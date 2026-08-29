#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 3 ]; then
	printf 'usage: %s PHASE neo4j-4.4.48|neo4j-5.26.30 NEW_OUTPUT_DIRECTORY\n' "$0" >&2
	exit 2
fi

phase=$1
source_version=$2
output=$3
require_phase_approval "$phase"
require_command agefreighter
require_command psql

case "$source_version" in
	neo4j-4.4.48)
		require_nonempty_environment AGEFREIGHTER_TARGET_DSN_NEO4J44
		target_dsn=$AGEFREIGHTER_TARGET_DSN_NEO4J44
		;;
	neo4j-5.26.30)
		require_nonempty_environment AGEFREIGHTER_TARGET_DSN_NEO4J526
		target_dsn=$AGEFREIGHTER_TARGET_DSN_NEO4J526
		;;
	*)
		printf 'unsupported source version: %s\n' "$source_version" >&2
		exit 2
		;;
esac

config="$(simulation_root)/configs/$source_version.yaml"
require_file "$config"
if [ -e "$output" ]; then
	printf 'output path already exists: %s\n' "$output" >&2
	exit 3
fi
mkdir -p "$output"

agefreighter optimize --target "$config" --format json \
	--output "$output/optimize-before.json"
psql "$target_dsn" -X --set ON_ERROR_STOP=1 --command 'ANALYZE' \
	>"$output/analyze.txt"
agefreighter optimize --target "$config" --format json \
	--output "$output/optimize-after.json"

printf 'post-load statistics refreshed; review optimization reports under %s\n' "$output"
unset target_dsn
