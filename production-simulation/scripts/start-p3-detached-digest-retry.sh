#!/bin/sh

set -eu

if [ "$(id -u)" -ne 0 ] || [ "$#" -ne 5 ]; then
	printf 'usage as root: %s neo4j-4.4.48|neo4j-5.26.30 RUN_ID TARGET_DATABASE SOURCE_RUN_ID JOB_ID\n' "$0" >&2
	exit 2
fi

source_version=$1
run_id=$2
target_database=$3
source_run_id=$4
job_id=$5
case "$source_version" in
	neo4j-4.4.48|neo4j-5.26.30) ;;
	*) printf 'unsupported source version: %s\n' "$source_version" >&2; exit 2 ;;
esac
for value in "$run_id" "$target_database" "$source_run_id"; do
	case "$value" in
		*[!a-zA-Z0-9_.-]*|'') printf 'run or database value contains unsupported characters\n' >&2; exit 2 ;;
	esac
done
case "$job_id" in
	????????-????-????-????-????????????) ;;
	*) printf 'JOB_ID must be a UUID\n' >&2; exit 2 ;;
esac

: "${KEY_VAULT_NAME:?KEY_VAULT_NAME is required}"
: "${POSTGRES_FQDN:?POSTGRES_FQDN is required}"
: "${PRODUCTION_SIMULATION_GIT_REF:?PRODUCTION_SIMULATION_GIT_REF is required}"

if systemctl list-units --type=service --state=running 'agefreighter-p3-*' \
		--no-legend --no-pager | grep -q .; then
	printf 'another P3 unit is active\n' >&2
	exit 3
fi

unit_suffix=$(printf '%s' "$run_id" | tr '._' '--')
unit="agefreighter-p3-$unit_suffix"
control_root=/var/lib/agefreighter-p3/control
mkdir -p "$control_root"
log="$control_root/$run_id.log"
if [ -e "$log" ]; then
	printf 'control log already exists: %s\n' "$log" >&2
	exit 3
fi

runner=/opt/agefreighter-production-simulation/production-simulation/scripts/run-p3-target-digest-retry-vm.sh
systemd-run \
	--unit "$unit" \
	--property "StandardOutput=append:$log" \
	--property "StandardError=append:$log" \
	--setenv "KEY_VAULT_NAME=$KEY_VAULT_NAME" \
	--setenv "POSTGRES_FQDN=$POSTGRES_FQDN" \
	--setenv "PRODUCTION_SIMULATION_GIT_REF=$PRODUCTION_SIMULATION_GIT_REF" \
	"$runner" "$source_version" "$run_id" "$target_database" \
		"$source_run_id" "$job_id"

printf 'started detached unit %s; evidence log %s\n' "$unit" "$log"
