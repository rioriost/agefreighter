#!/bin/sh

set -eu

if [ "$(id -u)" -ne 0 ] || { [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; }; then
	printf 'usage as root: %s [p0|p1|p2] neo4j-4.4.48|neo4j-5.26.30\n' "$0" >&2
	exit 2
fi

if [ "$#" -eq 1 ]; then
	phase=p0
	source_version=$1
else
	phase=$1
	source_version=$2
fi
case "$phase" in
	p0|p1|p2) ;;
	*)
		printf 'unsupported loader phase: %s\n' "$phase" >&2
		exit 2
		;;
esac
export GOPATH=/root/go
export GOMODCACHE=/root/go/pkg/mod
export GOCACHE=/root/.cache/agefreighter-go-build
mkdir -p "$GOMODCACHE" "$GOCACHE"
target_database_prefix=${TARGET_DATABASE_PREFIX:-agefreighter_$phase}
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

state_root="/var/lib/agefreighter-$phase"
mkdir -p /opt/agefreighter/bin "$state_root/results"
cd "$source_root"
go build -trimpath -o /opt/agefreighter/bin/agefreighter ./cmd/agefreighter
go build -trimpath -o /opt/agefreighter/bin/rangedigest \
	./production-simulation/cmd/rangedigest
go build -trimpath -o /opt/agefreighter/bin/fixturegen \
	./production-simulation/cmd/fixturegen

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
pgpass_file="/run/agefreighter-$phase.pgpass"
escaped_postgres_password=$(printf '%s' "$postgres_password" |
	sed 's/\\/\\\\/g; s/:/\\:/g')
umask 077
printf '%s:%s:%s:%s:%s\n' "$POSTGRES_FQDN" 5432 '*' \
	agefreighter "$escaped_postgres_password" >"$pgpass_file"
export PGPASSFILE="$pgpass_file"
trap 'rm -f "$pgpass_file"' EXIT HUP INT TERM
case "${PREPARE_TARGET_DATABASES:-0}" in
	0) ;;
	1)
		AGEFREIGHTER_ADMIN_DSN="host=$POSTGRES_FQDN port=5432 dbname=postgres user=agefreighter sslmode=require"
		export AGEFREIGHTER_ADMIN_DSN TARGET_DATABASE_PREFIX
		PRODUCTION_SIMULATION_APPROVAL="reviewed-$phase" \
			"$source_root/production-simulation/scripts/prepare-target-databases.sh" "$phase"
		unset AGEFREIGHTER_ADMIN_DSN
		;;
	*)
		printf 'PREPARE_TARGET_DATABASES must be 0 or 1\n' >&2
		exit 2
		;;
esac
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
AGEFREIGHTER_TARGET_DSN=$target_dsn \
	PRODUCTION_SIMULATION_APPROVAL="reviewed-$phase" \
	"$source_root/production-simulation/scripts/preflight-target.sh" "$phase"
run_id=${PRODUCTION_SIMULATION_RUN_ID:-$source_version}
case "$run_id" in
	*[!a-z0-9_.-]*|'')
		printf 'PRODUCTION_SIMULATION_RUN_ID contains unsupported characters\n' >&2
		exit 2
		;;
esac
result_root="$state_root/results/$run_id/$source_version"
if [ -e "$result_root" ]; then
	printf 'result path already exists: %s\n' "$result_root" >&2
	exit 3
fi
mkdir -p "$result_root"

verification_level=${P2_VERIFICATION_LEVEL:-full}
resume_job_id=${P2_RESUME_JOB_ID:-}
run_source_profiles=0
run_canonical_digest=0
case "$verification_level" in
	full)
		run_source_profiles=1
		run_canonical_digest=1
		;;
	digest)
		if [ "$phase" != p2 ]; then
			printf 'digest verification is supported only for P2\n' >&2
			exit 2
		fi
		run_canonical_digest=1
		;;
	tuning)
		if [ "$phase" != p2 ]; then
			printf 'tuning verification is supported only for P2\n' >&2
			exit 2
		fi
		;;
	*)
		printf 'P2_VERIFICATION_LEVEL must be full, digest, or tuning\n' >&2
		exit 2
		;;
esac

if [ -n "$resume_job_id" ]; then
	if [ "$phase" != p2 ]; then
		printf 'P2_RESUME_JOB_ID is supported only for P2\n' >&2
		exit 2
	fi
	case "$resume_job_id" in
		????????-????-????-????-????????????) ;;
		*) printf 'P2_RESUME_JOB_ID must be a UUID\n' >&2; exit 2 ;;
	esac
fi

if [ "$phase" = p2 ]; then
	config="$result_root/job.yaml"
	cp "$source_root/production-simulation/configs/$source_version.yaml" "$config"
	case "${P2_FETCH_ROWS:-5000}" in 5000|10000) ;; *) printf 'unsupported P2_FETCH_ROWS\n' >&2; exit 2 ;; esac
	case "${P2_BATCH_ROWS:-20000}" in 10000|20000) ;; *) printf 'unsupported P2_BATCH_ROWS\n' >&2; exit 2 ;; esac
	case "${P2_BATCH_BYTES:-64MiB}" in 64MiB|128MiB) ;; *) printf 'unsupported P2_BATCH_BYTES\n' >&2; exit 2 ;; esac
	case "${P2_TARGET_MODE:-create}" in create|replace) ;; *) printf 'unsupported P2_TARGET_MODE\n' >&2; exit 2 ;; esac
	sed -i \
		-e "s/fetchRows: 5000/fetchRows: ${P2_FETCH_ROWS:-5000}/" \
		-e "s/batchRows: 20000/batchRows: ${P2_BATCH_ROWS:-20000}/" \
		-e "s/batchBytes: 64MiB/batchBytes: ${P2_BATCH_BYTES:-64MiB}/" \
		-e "s/mode: create/mode: ${P2_TARGET_MODE:-create}/" \
		"$config"
fi

{
	printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	uname -a
	lscpu
	free -b
	df -B1
} >"$result_root/host-before.txt"
/opt/agefreighter/bin/agefreighter validate "$config" >"$result_root/validate.txt"
/opt/agefreighter/bin/agefreighter plan "$config" >"$result_root/plan.json"
if [ "$run_source_profiles" -eq 1 ]; then
	/opt/agefreighter/bin/agefreighter profile --mode exact --format json "$config" \
		>"$result_root/source-profile-before.json"
fi
if [ -n "$resume_job_id" ]; then
	printf '%s\n' "$resume_job_id" >"$result_root/resumed-job-id.txt"
	/usr/bin/time -v -o "$result_root/time.txt" \
		/opt/agefreighter/bin/agefreighter resume --job "$config" "$resume_job_id" \
		>"$result_root/load.json" 2>"$result_root/load.stderr"
else
	/usr/bin/time -v -o "$result_root/time.txt" \
		/opt/agefreighter/bin/agefreighter load "$config" \
		>"$result_root/load.json" 2>"$result_root/load.stderr"
fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$result_root/completed-at.txt"

job_id=$(jq -er '.jobId' "$result_root/load.json")
printf '%s\n' "$job_id" >"$result_root/job-id.txt"
/opt/agefreighter/bin/agefreighter report --target "$config" --include-counts \
	--format json --output "$result_root/report.json" "$job_id"
/opt/agefreighter/bin/agefreighter verify --target "$config" --counts --integrity \
	--limit 1000 --format json --output "$result_root/verify.json" "$job_id"
if [ "$run_source_profiles" -eq 1 ]; then
	/opt/agefreighter/bin/agefreighter profile --mode exact --format json "$config" \
		>"$result_root/source-profile-after.json"
fi
/opt/agefreighter/bin/agefreighter doctor --target "$config" --format json \
	--output "$result_root/doctor.json"
/opt/agefreighter/bin/agefreighter optimize --target "$config" --format json \
	--output "$result_root/optimize.json"

/opt/agefreighter/bin/agefreighter optimize --target "$config" --format json \
	--output "$result_root/optimize-before-analyze.json"
psql "$target_dsn" -X --set ON_ERROR_STOP=1 --command 'ANALYZE' \
	>"$result_root/analyze.txt"
/opt/agefreighter/bin/agefreighter optimize --target "$config" --format json \
	--output "$result_root/optimize-after-analyze.json"

if [ "$run_canonical_digest" -eq 1 ]; then
	fixture_root="$state_root/fixture-$phase"
	if [ ! -f "$fixture_root/manifest.json" ]; then
		/opt/agefreighter/bin/fixturegen generate --phase "$phase" \
			--output "$fixture_root" --shards 64 --workers 8 --seed 20260829
	fi
	/opt/agefreighter/bin/fixturegen verify --manifest "$fixture_root/manifest.json"
	/opt/agefreighter/bin/rangedigest fixture --manifest "$fixture_root/manifest.json" \
		--range-rows 100000 --output "$result_root/fixture-digest.json"
	case "$source_version" in
		neo4j-4.4.48) target_environment=AGEFREIGHTER_TARGET_DSN_NEO4J44 ;;
		neo4j-5.26.30) target_environment=AGEFREIGHTER_TARGET_DSN_NEO4J526 ;;
	esac
	/opt/agefreighter/bin/rangedigest target --manifest "$fixture_root/manifest.json" \
		--job-id "$job_id" --dsn-env "$target_environment" --range-rows 100000 \
		--output "$result_root/target-digest.json"
	/opt/agefreighter/bin/rangedigest compare \
		--expected "$result_root/fixture-digest.json" \
		--actual "$result_root/target-digest.json" \
		--output "$result_root/digest-comparison.json"
else
	printf '%s\n' '{"status":"not-run","reason":"P2 tuning candidate; full digest is required for the baseline and frozen configuration"}' \
		>"$result_root/digest-comparison.json"
fi

jq -n \
	--arg source "$source_version" \
	--arg jobId "$job_id" \
	--arg resumedJobId "$resume_job_id" \
	--arg commit "$(git -C "$source_root" rev-parse HEAD)" \
	--argjson load "$(cat "$result_root/load.json")" \
	--argjson verification "$(cat "$result_root/verify.json")" \
	--argjson digest "$(cat "$result_root/digest-comparison.json")" \
	'{source:$source,jobId:$jobId,resumedJobId:$resumedJobId,commit:$commit,load:$load,verification:$verification,digest:$digest}' \
	>"$result_root/summary.json"

jq '{source,jobId,commit,loadStatus:.load.status,verificationOutcome:.verification.outcome,digestStatus:.digest.status}' \
	"$result_root/summary.json"

unset AGEFREIGHTER_NEO4J_PASSWORD AGEFREIGHTER_TARGET_DSN_NEO4J44
unset AGEFREIGHTER_TARGET_DSN_NEO4J526 target_dsn
unset neo4j_password postgres_password token
rm -f "$pgpass_file"
trap - EXIT HUP INT TERM
