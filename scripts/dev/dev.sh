#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/services.sh"
. "$SCRIPT_DIR/runtime.sh"

READY_TIMEOUT=${DEV_READY_TIMEOUT:-90}

validate_resources() {
	for name in $(service_containers) $(service_volumes); do
		if ! validate_resource_name "$name"; then
			printf 'Unsafe development resource name: %s\n' "$name" >&2
			exit 1
		fi
	done
}

ensure_volume() {
	if ! runtime_has_volume "$1"; then
		printf 'Creating volume %s\n' "$1"
		runtime_create_volume "$1"
	fi
}

ensure_container() {
	name=$1
	image=$2
	config=$3
	shift 3
	if runtime_has_container "$name"; then
		if ! runtime_is_managed "$name"; then
			printf 'Refusing unmanaged container with reserved name: %s\n' "$name" >&2
			return 1
		fi
		if ! runtime_matches_image "$name" "$image"; then
			printf 'Container %s uses a stale image; run make dev-reset\n' "$name" >&2
			return 1
		fi
		if ! runtime_matches_config "$name" "$config"; then
			printf 'Recreating container %s for updated configuration\n' "$name"
			runtime_delete "$name"
		elif ! runtime_is_running "$name"; then
			printf 'Starting container %s\n' "$name"
			runtime_start "$name"
			return 0
		else
			return 0
		fi
	fi
	printf 'Creating container %s\n' "$name"
	runtime_run "$name" \
		--label "io.agefreighter.config=$config" \
		"$@" "$image"
}

start_age() {
	ensure_volume "$AGE_VOLUME"
	ensure_container "$AGE_CONTAINER" "$AGE_IMAGE" "$AGE_CONTAINER_CONFIG" \
		--publish "127.0.0.1:$AGE_PORT:5432" \
		--memory 1G \
		--volume "$AGE_VOLUME:/var/lib/postgresql/data" \
		--env PGDATA=/var/lib/postgresql/data/pgdata \
		--env POSTGRES_USER=agefreighter \
		--env "POSTGRES_PASSWORD=$AGEFREIGHTER_DEV_PASSWORD" \
		--env POSTGRES_DB=agefreighter
}

start_postgres() {
	ensure_volume "$POSTGRES_VOLUME"
	ensure_container "$POSTGRES_CONTAINER" "$POSTGRES_IMAGE" "$POSTGRES_CONTAINER_CONFIG" \
		--publish "127.0.0.1:$POSTGRES_PORT:5432" \
		--volume "$POSTGRES_VOLUME:/var/lib/postgresql/data" \
		--env PGDATA=/var/lib/postgresql/data/pgdata \
		--env POSTGRES_USER=agefreighter \
		--env "POSTGRES_PASSWORD=$AGEFREIGHTER_DEV_PASSWORD" \
		--env POSTGRES_DB=agefreighter
}

start_neo4j() {
	ensure_volume "$NEO4J_VOLUME"
	ensure_container "$NEO4J_CONTAINER" "$NEO4J_IMAGE" "$NEO4J_CONTAINER_CONFIG" \
		--publish "127.0.0.1:$NEO4J_BOLT_PORT:7687" \
		--publish "127.0.0.1:$NEO4J_HTTP_PORT:7474" \
		--volume "$NEO4J_VOLUME:/data" \
		--env "NEO4J_AUTH=neo4j/$AGEFREIGHTER_DEV_PASSWORD" \
		--env NEO4J_server_memory_heap_initial__size=256m \
		--env NEO4J_server_memory_heap_max__size=512m \
		--env NEO4J_server_memory_pagecache_size=256m
}

wait_ready() {
	name=$1
	shift
	started=$(date +%s)
	while ! runtime_exec "$name" "$@" >/dev/null 2>&1; do
		now=$(date +%s)
		if [ $((now - started)) -ge "$READY_TIMEOUT" ]; then
			printf 'Timed out waiting for %s after %ss\n' "$name" "$READY_TIMEOUT" >&2
			runtime_logs "$name" >&2 || true
			return 1
		fi
		sleep 1
	done
}

wait_services() {
	wait_ready "$AGE_CONTAINER" \
		pg_isready -U agefreighter -d agefreighter
	wait_ready "$POSTGRES_CONTAINER" \
		pg_isready -U agefreighter -d agefreighter
	wait_ready "$NEO4J_CONTAINER" \
		cypher-shell -u neo4j -p "$AGEFREIGHTER_DEV_PASSWORD" 'RETURN 1'
}

initialize_fixtures() {
	runtime_exec "$AGE_CONTAINER" \
		psql -v ON_ERROR_STOP=1 -U agefreighter -d agefreighter \
		-c 'CREATE EXTENSION IF NOT EXISTS age; LOAD '\''age'\''; SET search_path = ag_catalog, "$user", public; SELECT create_graph('\''fixture_graph'\'') WHERE NOT EXISTS (SELECT 1 FROM ag_graph WHERE name = '\''fixture_graph'\'');' \
		>/dev/null

	runtime_exec "$POSTGRES_CONTAINER" \
		psql -v ON_ERROR_STOP=1 -U agefreighter -d agefreighter \
		-c 'CREATE TABLE IF NOT EXISTS people (person_id bigint PRIMARY KEY, full_name text NOT NULL); CREATE TABLE IF NOT EXISTS knows (relationship_id bigint PRIMARY KEY, from_id bigint NOT NULL REFERENCES people(person_id), to_id bigint NOT NULL REFERENCES people(person_id)); TRUNCATE knows, people; INSERT INTO people VALUES (1, '\''Ada'\''), (2, '\''Grace'\''); INSERT INTO knows VALUES (1, 1, 2);' \
		>/dev/null

	runtime_exec "$NEO4J_CONTAINER" \
		cypher-shell -u neo4j -p "$AGEFREIGHTER_DEV_PASSWORD" \
		'CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE; MERGE (:Person {id: 1, name: "Ada"}); MERGE (:Person {id: 2, name: "Grace"});' \
		>/dev/null
}

smoke() {
	runtime_exec "$AGE_CONTAINER" \
		psql -v ON_ERROR_STOP=1 -U agefreighter -d agefreighter \
		-tAc "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = 'fixture_graph'" |
		grep -Fqx 1
	runtime_exec "$POSTGRES_CONTAINER" \
		psql -v ON_ERROR_STOP=1 -U agefreighter -d agefreighter \
		-tAc 'SELECT (SELECT count(*) FROM people), (SELECT count(*) FROM knows)' |
		grep -Fqx '2|1'
	runtime_exec "$NEO4J_CONTAINER" \
		cypher-shell --format plain -u neo4j -p "$AGEFREIGHTER_DEV_PASSWORD" \
		'MATCH (p:Person) RETURN count(p) AS count' |
		grep -Eq '(^|[[:space:]])2([[:space:]]|$)'
	printf 'Development database smoke checks passed\n'
}

pull() {
	for image in $(service_images); do
		printf 'Pulling %s for %s\n' "$image" "$PLATFORM"
		runtime_pull "$image"
	done
}

up() {
	start_age
	start_postgres
	start_neo4j
	wait_services
	initialize_fixtures
	smoke
}

status() {
	for name in $(service_containers); do
		if runtime_is_running "$name"; then
			printf '%-38s running\n' "$name"
		elif runtime_has_container "$name"; then
			printf '%-38s stopped\n' "$name"
		else
			printf '%-38s absent\n' "$name"
		fi
	done
}

down() {
	for name in $(service_containers); do
		if runtime_is_running "$name"; then
			printf 'Stopping container %s\n' "$name"
			runtime_stop "$name"
		fi
	done
}

reset() {
	for name in $(service_containers); do
		if runtime_has_container "$name"; then
			if ! runtime_is_managed "$name"; then
				printf 'Refusing to delete unmanaged container: %s\n' "$name" >&2
				return 1
			fi
			printf 'Deleting container %s\n' "$name"
			runtime_delete "$name"
		fi
	done
	for volume in $(service_volumes); do
		if runtime_has_volume "$volume"; then
			printf 'Deleting volume %s\n' "$volume"
			runtime_delete_volume "$volume"
		fi
	done
	up
}

usage() {
	printf 'Usage: %s {pull|up|status|smoke|down|reset}\n' "$0" >&2
	exit 2
}

validate_resources
case "$READY_TIMEOUT" in
	'' | *[!0-9]* | 0)
		printf 'DEV_READY_TIMEOUT must be a positive integer\n' >&2
		exit 1
		;;
esac
detect_runtime
ensure_runtime

case "${1:-}" in
	pull)
		pull
		;;
	up)
		up
		;;
	status)
		status
		;;
	smoke)
		smoke
		;;
	down)
		down
		;;
	reset)
		reset
		;;
	*)
		usage
		;;
esac
