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
	neo4j-4.4.48) target_environment=AGEFREIGHTER_TARGET_DSN_NEO4J44 ;;
	neo4j-5.26.30) target_environment=AGEFREIGHTER_TARGET_DSN_NEO4J526 ;;
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

source_root=/opt/agefreighter-production-simulation
state_root=/var/lib/agefreighter-p3
result_root="$state_root/results/$run_id/$source_version"
source_result_root="$state_root/results/$source_run_id/$source_version"
fixture_root="$state_root/fixture-p3"

if [ -e "$result_root" ]; then
	printf 'result path already exists: %s\n' "$result_root" >&2
	exit 3
fi
if [ ! -f "$source_result_root/load.json" ] || \
		[ ! -f "$source_result_root/job-id.txt" ] || \
		[ ! -f "$source_result_root/fixture-digest.json" ]; then
	printf 'source run evidence is incomplete: %s\n' "$source_result_root" >&2
	exit 3
fi
if [ "$(cat "$source_result_root/job-id.txt")" != "$job_id" ] || \
		[ "$(jq -r '.status' "$source_result_root/load.json")" != committed ]; then
	printf 'source run does not identify the requested committed job\n' >&2
	exit 3
fi
if [ ! -f "$fixture_root/manifest.json" ]; then
	printf 'P3 fixture manifest is missing\n' >&2
	exit 3
fi

git -C "$source_root" fetch -q origin "$PRODUCTION_SIMULATION_GIT_REF"
git -C "$source_root" checkout -q --detach "$PRODUCTION_SIMULATION_GIT_REF"
go -C "$source_root" build -trimpath -o /opt/agefreighter/bin/rangedigest \
	./production-simulation/cmd/rangedigest

mkdir -p "$result_root"
printf '%s\n' "$job_id" >"$result_root/job-id.txt"
printf '%s\n' "$source_run_id" >"$result_root/source-run-id.txt"
git -C "$source_root" rev-parse HEAD >"$result_root/verifier-commit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$result_root/started-at.txt"
cp "$source_result_root/fixture-digest.json" "$result_root/fixture-digest.json"

token=$(curl -fsS -H Metadata:true \
	'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net' |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
postgres_password=$(curl -fsS -H "Authorization: Bearer $token" \
	"https://$KEY_VAULT_NAME.vault.azure.net/secrets/postgres-admin-password?api-version=7.4" |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["value"])')
pgpass_file="/run/agefreighter-p3-$run_id.pgpass"
escaped_postgres_password=$(printf '%s' "$postgres_password" |
	sed 's/\\/\\\\/g; s/:/\\:/g')
umask 077
printf '%s:%s:%s:%s:%s\n' "$POSTGRES_FQDN" 5432 '*' \
	agefreighter "$escaped_postgres_password" >"$pgpass_file"
export PGPASSFILE="$pgpass_file"
trap 'rm -f "$pgpass_file"' EXIT HUP INT TERM

target_dsn="host=$POSTGRES_FQDN port=5432 dbname=$target_database user=agefreighter sslmode=require"
case "$target_environment" in
	AGEFREIGHTER_TARGET_DSN_NEO4J44)
		AGEFREIGHTER_TARGET_DSN_NEO4J44=$target_dsn
		export AGEFREIGHTER_TARGET_DSN_NEO4J44
		;;
	AGEFREIGHTER_TARGET_DSN_NEO4J526)
		AGEFREIGHTER_TARGET_DSN_NEO4J526=$target_dsn
		export AGEFREIGHTER_TARGET_DSN_NEO4J526
		;;
esac

target_temporary="$result_root/target-digest.json.tmp"
/usr/bin/time -v -o "$result_root/target-digest-time.txt" \
	/opt/agefreighter/bin/rangedigest target \
		--manifest "$fixture_root/manifest.json" \
		--job-id "$job_id" \
		--dsn-env "$target_environment" \
		--range-rows 100000 \
		--output "$target_temporary"
mv "$target_temporary" "$result_root/target-digest.json"
/opt/agefreighter/bin/rangedigest compare \
	--expected "$result_root/fixture-digest.json" \
	--actual "$result_root/target-digest.json" \
	--output "$result_root/digest-comparison.json"
date -u +%Y-%m-%dT%H:%M:%SZ >"$result_root/completed-at.txt"

jq -n \
	--arg source "$source_version" \
	--arg sourceRun "$source_run_id" \
	--arg jobId "$job_id" \
	--arg verifierCommit "$(cat "$result_root/verifier-commit.txt")" \
	--argjson digest "$(cat "$result_root/digest-comparison.json")" \
	'{source:$source,sourceRun:$sourceRun,jobId:$jobId,verifierCommit:$verifierCommit,digest:$digest}' \
	>"$result_root/summary.json"
jq '{source,sourceRun,jobId,verifierCommit,digestStatus:.digest.status}' \
	"$result_root/summary.json"

unset AGEFREIGHTER_TARGET_DSN_NEO4J44 AGEFREIGHTER_TARGET_DSN_NEO4J526
unset target_dsn
unset postgres_password token
rm -f "$pgpass_file"
trap - EXIT HUP INT TERM
