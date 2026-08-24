#!/bin/sh

detect_runtime() {
	if [ "${DEV_RUNTIME:-auto}" = "apple" ]; then
		RUNTIME=apple
		RUNTIME_CLI=${CONTAINER_CLI:-container}
	elif [ "${DEV_RUNTIME:-auto}" = "docker" ]; then
		RUNTIME=docker
		RUNTIME_CLI=${DOCKER_CLI:-docker}
	elif [ "$(uname -s)" = "Darwin" ] && command -v container >/dev/null 2>&1; then
		RUNTIME=apple
		RUNTIME_CLI=${CONTAINER_CLI:-container}
	elif command -v docker >/dev/null 2>&1; then
		RUNTIME=docker
		RUNTIME_CLI=${DOCKER_CLI:-docker}
	else
		printf 'Apple Container or Docker is required\n' >&2
		return 1
	fi

	if ! command -v "$RUNTIME_CLI" >/dev/null 2>&1; then
		printf '%s CLI not found: %s\n' "$RUNTIME" "$RUNTIME_CLI" >&2
		return 1
	fi

	if [ -n "${DEV_PLATFORM:-}" ]; then
		PLATFORM=$DEV_PLATFORM
	elif [ "$RUNTIME" = "apple" ]; then
		PLATFORM=linux/arm64
	else
		PLATFORM=linux/amd64
	fi
	export RUNTIME RUNTIME_CLI PLATFORM
}

ensure_runtime() {
	if [ "$RUNTIME" = "apple" ]; then
		if ! "$RUNTIME_CLI" system status 2>/dev/null |
			awk '$1 == "status" && $2 == "running" { found = 1 } END { exit !found }'
		then
			printf 'Starting Apple Container services\n'
			"$RUNTIME_CLI" system start --disable-kernel-install --timeout 60
		fi
	else
		"$RUNTIME_CLI" info >/dev/null
	fi
}

runtime_pull() {
	if [ "$RUNTIME" = "apple" ]; then
		"$RUNTIME_CLI" image pull --platform "$PLATFORM" "$1"
	else
		"$RUNTIME_CLI" pull --platform "$PLATFORM" "$1"
	fi
}

runtime_container_names() {
	if [ "$RUNTIME" = "apple" ]; then
		"$RUNTIME_CLI" list --all --quiet
	else
		"$RUNTIME_CLI" ps --all --format '{{.Names}}'
	fi
}

runtime_running_names() {
	if [ "$RUNTIME" = "apple" ]; then
		"$RUNTIME_CLI" list --quiet
	else
		"$RUNTIME_CLI" ps --format '{{.Names}}'
	fi
}

runtime_volume_names() {
	if [ "$RUNTIME" = "apple" ]; then
		"$RUNTIME_CLI" volume list --quiet
	else
		"$RUNTIME_CLI" volume ls --format '{{.Name}}'
	fi
}

runtime_has_container() {
	runtime_container_names | grep -Fqx "$1"
}

runtime_is_running() {
	runtime_running_names | grep -Fqx "$1"
}

runtime_has_volume() {
	runtime_volume_names | grep -Fqx "$1"
}

runtime_is_managed() {
	"$RUNTIME_CLI" inspect "$1" 2>/dev/null |
		grep -Eq '"io.agefreighter.managed"[[:space:]]*:[[:space:]]*"true"'
}

runtime_matches_image() {
	name=$1
	image=$2
	digest=${image##*@}
	"$RUNTIME_CLI" inspect "$name" 2>/dev/null | grep -Fq "$digest"
}

runtime_create_volume() {
	"$RUNTIME_CLI" volume create "$1" >/dev/null
}

runtime_run() {
	name=$1
	shift
	"$RUNTIME_CLI" run --detach \
		--name "$name" \
		--platform "$PLATFORM" \
		--label io.agefreighter.managed=true \
		"$@" >/dev/null
}

runtime_start() {
	"$RUNTIME_CLI" start "$1" >/dev/null
}

runtime_stop() {
	"$RUNTIME_CLI" stop "$1" >/dev/null
}

runtime_delete() {
	if [ "$RUNTIME" = "apple" ]; then
		"$RUNTIME_CLI" delete --force "$1" >/dev/null
	else
		"$RUNTIME_CLI" rm --force "$1" >/dev/null
	fi
}

runtime_delete_volume() {
	if [ "$RUNTIME" = "apple" ]; then
		"$RUNTIME_CLI" volume delete "$1" >/dev/null
	else
		"$RUNTIME_CLI" volume rm "$1" >/dev/null
	fi
}

runtime_exec() {
	name=$1
	shift
	"$RUNTIME_CLI" exec "$name" "$@"
}

runtime_logs() {
	"$RUNTIME_CLI" logs "$1"
}
