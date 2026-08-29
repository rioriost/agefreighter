#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 3 ]; then
	printf 'usage: %s PHASE CONFIG JOB_ID\n' "$0" >&2
	exit 2
fi

phase=$1
config=$2
job_id=$3
require_phase_approval "$phase"
require_command agefreighter
require_command jq
require_file "$config"
require_nonempty_environment AGEFREIGHTER_TARGET_DSN

case "$job_id" in
	????????-????-????-????-????????????) ;;
	*)
		printf 'JOB_ID is not in UUID form\n' >&2
	exit 2
		;;
esac

raw_root="$(simulation_root)/results/raw/$job_id"
if [ -e "$raw_root" ]; then
	printf 'result directory already exists: %s\n' "$raw_root" >&2
	exit 3
fi
mkdir -p "$raw_root"

agefreighter report --target "$config" --include-counts --format json \
	--output "$raw_root/report.json" "$job_id"
agefreighter verify --target "$config" --counts --integrity --format json \
	--output "$raw_root/verify.json" "$job_id"
agefreighter doctor --target "$config" --format json \
	--output "$raw_root/doctor.json"
agefreighter optimize --target "$config" --format json \
	--output "$raw_root/optimize.json"

printf 'Raw results collected under %s; redact before creating a tracked summary.\n' "$raw_root"
