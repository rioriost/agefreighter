#!/bin/sh

set -eu

if [ "$(id -u)" -ne 0 ] || [ "$#" -ne 1 ]; then
	printf 'usage as root: %s neo4j-4.4.48|neo4j-5.26.30\n' "$0" >&2
	exit 2
fi

source_version=$1
export HOME=/root
export GOPATH=/root/go
export GOMODCACHE=/root/go/pkg/mod
mkdir -p "$GOMODCACHE"
target_database_prefix=${TARGET_DATABASE_PREFIX:-agefreighter_p0}
case "$target_database_prefix" in
	*[!a-z0-9_]*|'')
		printf 'TARGET_DATABASE_PREFIX must contain only lowercase letters, digits, and underscores\n' >&2
		exit 2
		;;
esac
case "$source_version" in
	neo4j-4.4.48)
	target_database="${target_database_prefix}_neo4j44"
	;;
	neo4j-5.26.30)
	target_database="${target_database_prefix}_neo4j526"
	;;
	*)
		printf 'unsupported source version: %s\n' "$source_version" >&2
		exit 2
		;;
esac

: "${KEY_VAULT_NAME:?KEY_VAULT_NAME is required}"
: "${POSTGRES_FQDN:?POSTGRES_FQDN is required}"
: "${PRODUCTION_SIMULATION_GIT_REF:?PRODUCTION_SIMULATION_GIT_REF is required}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git golang-go jq postgresql-client time >/dev/null

source_root=/opt/agefreighter-production-simulation
if [ ! -d "$source_root/.git" ]; then
	git clone -q https://github.com/rioriost/agefreighter.git "$source_root"
fi
git -C "$source_root" fetch -q origin "$PRODUCTION_SIMULATION_GIT_REF"
git -C "$source_root" checkout -q --detach "$PRODUCTION_SIMULATION_GIT_REF"

mkdir -p /opt/agefreighter/bin /var/lib/agefreighter-p0/results
cd "$source_root"
go build -trimpath -o /opt/agefreighter/bin/agefreighter ./cmd/agefreighter
go build -trimpath -o /opt/agefreighter/bin/rangedigest \
	./production-simulation/cmd/rangedigest

token=$(curl -fsS -H Metadata:true \
	'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net' |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
neo4j_password=$(curl -fsS -H "Authorization: Bearer $token" \
	"https://$KEY_VAULT_NAME.vault.azure.net/secrets/neo4j-password?api-version=7.4" |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["value"])')
postgres_password=$(curl -fsS -H "Authorization: Bearer $token" \
	"https://$KEY_VAULT_NAME.vault.azure.net/secrets/postgres-admin-password?api-version=7.4" |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["value"])')

export AGEFREIGHTER_NEO4J_PASSWORD="$neo4j_password"
pgpass_file=/run/agefreighter-p0.pgpass
escaped_postgres_password=$(printf '%s' "$postgres_password" |
	sed 's/\\/\\\\/g; s/:/\\:/g')
umask 077
printf '%s:%s:%s:%s:%s\n' "$POSTGRES_FQDN" 5432 "$target_database" \
	agefreighter "$escaped_postgres_password" >"$pgpass_file"
export PGPASSFILE=$pgpass_file
trap 'rm -f "$pgpass_file"' EXIT HUP INT TERM
target_dsn="host=$POSTGRES_FQDN port=5432 dbname=$target_database user=agefreighter sslmode=require"
case "$source_version" in
	neo4j-4.4.48)
		AGEFREIGHTER_TARGET_DSN_NEO4J44=$target_dsn
		export AGEFREIGHTER_TARGET_DSN_NEO4J44
		;;
	neo4j-5.26.30)
		AGEFREIGHTER_TARGET_DSN_NEO4J526=$target_dsn
		export AGEFREIGHTER_TARGET_DSN_NEO4J526
		;;
esac
export AGEFREIGHTER_LOG_FORMAT=json

config="$source_root/production-simulation/configs/$source_version.yaml"
result_root="/var/lib/agefreighter-p0/results/$source_version"
if [ -e "$result_root" ]; then
	printf 'result path already exists: %s\n' "$result_root" >&2
	exit 3
fi
mkdir -p "$result_root"

/opt/agefreighter/bin/agefreighter validate "$config" >"$result_root/validate.txt"
/opt/agefreighter/bin/agefreighter plan "$config" >"$result_root/plan.json"
/usr/bin/time -v -o "$result_root/time.txt" \
	/opt/agefreighter/bin/agefreighter load "$config" \
	>"$result_root/load.json" 2>"$result_root/load.stderr"

job_id=$(jq -er '.jobId' "$result_root/load.json")
printf '%s\n' "$job_id" >"$result_root/job-id.txt"
/opt/agefreighter/bin/agefreighter report --target "$config" --include-counts \
	--format json --output "$result_root/report.json" "$job_id"
/opt/agefreighter/bin/agefreighter verify --target "$config" --counts --integrity \
	--limit 1000 --format json --output "$result_root/verify.json" "$job_id"
/opt/agefreighter/bin/agefreighter doctor --target "$config" --format json \
	--output "$result_root/doctor.json"
/opt/agefreighter/bin/agefreighter optimize --target "$config" --format json \
	--output "$result_root/optimize.json"

jq -n \
	--arg source "$source_version" \
	--arg jobId "$job_id" \
	--arg commit "$(git -C "$source_root" rev-parse HEAD)" \
	--argjson load "$(cat "$result_root/load.json")" \
	--argjson verification "$(cat "$result_root/verify.json")" \
	'{source:$source,jobId:$jobId,commit:$commit,load:$load,verification:$verification}' \
	>"$result_root/summary.json"

jq '{source,jobId,commit,loadStatus:.load.status,verificationOutcome:.verification.outcome}' \
	"$result_root/summary.json"

unset AGEFREIGHTER_NEO4J_PASSWORD AGEFREIGHTER_TARGET_DSN_NEO4J44
unset AGEFREIGHTER_TARGET_DSN_NEO4J526 target_dsn
unset neo4j_password postgres_password token
rm -f "$pgpass_file"
trap - EXIT HUP INT TERM
