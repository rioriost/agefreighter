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
case "$source_version" in
	neo4j-4.4.48|neo4j-5.26.30) ;;
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
export AGEFREIGHTER_TARGET_DSN="host=$POSTGRES_FQDN port=5432 dbname=agefreighter_p0 user=agefreighter password='$postgres_password' sslmode=require"
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

jq '{source,jobId,commit,loadStatus:.load.status,verificationStatus:.verification.status}' \
	"$result_root/summary.json"

unset AGEFREIGHTER_NEO4J_PASSWORD AGEFREIGHTER_TARGET_DSN
unset neo4j_password postgres_password token
