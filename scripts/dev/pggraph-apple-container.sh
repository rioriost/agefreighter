#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$repo_root"

container_name=agefreighter-pg19-sqlpgq
image_name=docker.io/library/postgres:19beta3
image_digest=sha256:a48b19841e04b35b72a25e9a94314ac80546d32b5e2e3cd9279390cbd8a99572
database=agefreighter
password=agefreighter-dev-only

fail() {
	printf '%s\n' "$*" >&2
	exit 1
}

require_tools() {
	command -v container >/dev/null 2>&1 || fail "Apple Container is required"
	command -v jq >/dev/null 2>&1 || fail "jq is required"
}

container_exists() {
	container list --all --format json |
		jq -e --arg name "$container_name" 'any(.[]; .id == $name)' >/dev/null
}

container_state() {
	container inspect "$container_name" | jq -er '.[0].status.state'
}

verify_container_image() {
	actual_digest=$(container inspect "$container_name" |
		jq -er '.[0].configuration.image.descriptor.digest')
	[ "$actual_digest" = "$image_digest" ] || fail \
		"$container_name uses $actual_digest; expected $image_digest"
}

wait_ready() {
	count=0
	while [ "$count" -lt 60 ]; do
		if container exec "$container_name" pg_isready \
			-U postgres -d "$database" >/dev/null 2>&1; then
			return
		fi
		count=$((count + 1))
		sleep 1
	done
	fail "$container_name did not become ready"
}

up() {
	container image pull "$image_name"
	actual_digest=$(container image inspect "$image_name" |
		jq -er '.[0].configuration.descriptor.digest')
	[ "$actual_digest" = "$image_digest" ] || fail \
		"$image_name resolved to $actual_digest; review the expected digest before testing"

	if container_exists; then
		verify_container_image
		state=$(container_state)
		case "$state" in
			running) ;;
			stopped) container start "$container_name" >/dev/null ;;
			*) fail "$container_name is in unsupported state $state" ;;
		esac
	else
		container run --detach --name "$container_name" --memory 2g --cpus 4 \
			-e "POSTGRES_PASSWORD=$password" -e "POSTGRES_DB=$database" \
			"$image_name@$image_digest" >/dev/null
	fi
	wait_ready
}

status() {
	if ! container_exists; then
		printf '%s\n' "$container_name: absent"
		return
	fi
	verify_container_image
	state=$(container_state)
	printf '%s: %s\n' "$container_name" "$state"
	if [ "$state" != running ]; then
		return
	fi
	container exec "$container_name" psql -X -U postgres -d "$database" -Atc \
		'SELECT version(); SHOW server_version_num;'
}

test_target() {
	up
	address=$(container inspect "$container_name" |
		jq -er '.[0].status.networks[0].ipv4Address' | cut -d/ -f1)
	dsn="postgres://postgres:$password@$address:5432/$database?sslmode=disable"
	AGEFREIGHTER_PGGRAPH_TEST_DSN="$dsn" \
		go test -count=1 -v ./internal/pggraph ./internal/app \
		-run '^(TestPropertyGraphIntegration|TestPropertyGraphOperationGuardsIntegration|TestPropertyGraphMutationLockIntegration|TestPropertyGraphSinkReplayAndAbortIntegration|TestPropertyGraphSinkFailureIntegration|TestPostgreSQLPropertyGraphCreateAndResumeIntegration|TestPostgreSQLPropertyGraphModeMatrixIntegration|TestPostgreSQLPropertyGraphIncrementalResumeIntegration|TestPostgreSQLPropertyGraphReplaceRecoveryIntegration|TestPostgreSQLPropertyGraphIncrementalAdmissionIntegration|TestPostgreSQLPropertyGraphCorruptionDetectionIntegration)$$'
	AGEFREIGHTER_AGE_TEST_DSN="$dsn" \
		go test -count=1 -v ./internal/meta \
		-run '^TestMetadataV14V17V18V19V20UpgradeToV21Integration$$'
}

down() {
	if container_exists && [ "$(container_state)" = running ]; then
		container stop "$container_name" >/dev/null
	fi
}

require_tools
case "${1:-}" in
	up) up ;;
	status) status ;;
	test) test_target ;;
	down) down ;;
	*) fail "usage: $0 {up|status|test|down}" ;;
esac
