#!/bin/sh

set -eu

if [ "$(id -u)" -ne 0 ] || [ "$#" -ne 3 ]; then
	printf 'usage as root: %s neo4j-4.4.48|neo4j-5.26.30 RUN_ID TARGET_DATABASE_PREFIX\n' "$0" >&2
	exit 2
fi

source_version=$1
run_id=$2
target_database_prefix=$3
case "$source_version" in
	neo4j-4.4.48|neo4j-5.26.30) ;;
	*) printf 'unsupported source version: %s\n' "$source_version" >&2; exit 2 ;;
esac
case "$run_id" in
	*[!a-z0-9_.-]*|'') printf 'RUN_ID contains unsupported characters\n' >&2; exit 2 ;;
esac
case "$target_database_prefix" in
	*[!a-z0-9_]*|'') printf 'TARGET_DATABASE_PREFIX contains unsupported characters\n' >&2; exit 2 ;;
esac

: "${KEY_VAULT_NAME:?KEY_VAULT_NAME is required}"
: "${POSTGRES_FQDN:?POSTGRES_FQDN is required}"
: "${PRODUCTION_SIMULATION_GIT_REF:?PRODUCTION_SIMULATION_GIT_REF is required}"

case "${PREPARE_TARGET_DATABASES:-0}" in 0|1) ;; *) printf 'PREPARE_TARGET_DATABASES must be 0 or 1\n' >&2; exit 2 ;; esac
case "${P3_VERIFICATION_LEVEL:-full}" in full|digest) ;; *) printf 'invalid P3_VERIFICATION_LEVEL\n' >&2; exit 2 ;; esac
case "${P3_RESUME_JOB_ID:-}" in
	'') ;;
	????????-????-????-????-????????????) ;;
	*) printf 'P3_RESUME_JOB_ID must be empty or a UUID\n' >&2; exit 2 ;;
esac

unit_suffix=$(printf '%s' "$run_id" | tr '._' '--')
unit="agefreighter-p3-$unit_suffix"
control_root=/var/lib/agefreighter-p3/control
mkdir -p "$control_root"
log="$control_root/$run_id.log"
if [ -e "$log" ]; then
	printf 'control log already exists: %s\n' "$log" >&2
	exit 3
fi

runner=/opt/agefreighter-production-simulation/production-simulation/scripts/run-p0-loader-vm.sh
systemd-run \
	--unit "$unit" \
	--property "StandardOutput=append:$log" \
	--property "StandardError=append:$log" \
	--setenv "KEY_VAULT_NAME=$KEY_VAULT_NAME" \
	--setenv "POSTGRES_FQDN=$POSTGRES_FQDN" \
	--setenv "PRODUCTION_SIMULATION_GIT_REF=$PRODUCTION_SIMULATION_GIT_REF" \
	--setenv "PREPARE_TARGET_DATABASES=${PREPARE_TARGET_DATABASES:-0}" \
	--setenv "TARGET_DATABASE_PREFIX=$target_database_prefix" \
	--setenv "PRODUCTION_SIMULATION_RUN_ID=$run_id" \
	--setenv "P3_VERIFICATION_LEVEL=${P3_VERIFICATION_LEVEL:-full}" \
	--setenv "P3_RESUME_JOB_ID=${P3_RESUME_JOB_ID:-}" \
	"$runner" p3 "$source_version"

printf 'started detached unit %s; evidence log %s\n' "$unit" "$log"
