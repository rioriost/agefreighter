#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
	printf 'usage: %s PHASE NEW_OUTPUT_DIR [SHARDS] [WORKERS]\n' "$0" >&2
	exit 2
fi

phase=$1
output=$2
shards=${3:-64}
workers=${4:-8}

case "$phase" in
	tiny|p0|p1|p2|p3) ;;
	*)
		printf 'unsupported phase: %s\n' "$phase" >&2
		exit 2
		;;
esac

if [ -e "$output" ]; then
	printf 'output already exists; refusing to overwrite: %s\n' "$output" >&2
	exit 3
fi

require_command go
cd "$(repository_root)"
go run ./production-simulation/cmd/fixturegen generate \
	--phase "$phase" --output "$output" --shards "$shards" \
	--workers "$workers" --seed 20260829
go run ./production-simulation/cmd/fixturegen verify \
	--manifest "$output/manifest.json"
