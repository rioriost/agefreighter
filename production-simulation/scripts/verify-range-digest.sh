#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 5 ]; then
	printf 'usage: %s PHASE SOURCE_VERSION FIXTURE_MANIFEST JOB_ID NEW_OUTPUT_DIRECTORY\n' "$0" >&2
	exit 2
fi

phase=$1
source_version=$2
fixture_manifest=$3
job_id=$4
output=$5
require_phase_approval "$phase"
require_command rangedigest
require_file "$fixture_manifest"

case "$source_version" in
	neo4j-4.4.48)
		require_nonempty_environment AGEFREIGHTER_TARGET_DSN_NEO4J44
		target_environment=AGEFREIGHTER_TARGET_DSN_NEO4J44
		;;
	neo4j-5.26.30)
		require_nonempty_environment AGEFREIGHTER_TARGET_DSN_NEO4J526
		target_environment=AGEFREIGHTER_TARGET_DSN_NEO4J526
		;;
	*)
		printf 'unsupported source version: %s\n' "$source_version" >&2
		exit 2
		;;
esac
case "$job_id" in
	????????-????-????-????-????????????) ;;
	*)
		printf 'JOB_ID is not in UUID form\n' >&2
		exit 2
		;;
esac
if [ -e "$output" ]; then
	printf 'output path already exists: %s\n' "$output" >&2
	exit 3
fi
mkdir -p "$output"

rangedigest fixture --manifest "$fixture_manifest" --range-rows 100000 \
	--output "$output/fixture-digest.json"
rangedigest target --manifest "$fixture_manifest" --job-id "$job_id" \
	--dsn-env "$target_environment" --range-rows 100000 \
	--output "$output/target-digest.json"
rangedigest compare --expected "$output/fixture-digest.json" \
	--actual "$output/target-digest.json" --output "$output/comparison.json"

printf 'independent canonical range-digest verification passed: %s\n' "$output/comparison.json"
