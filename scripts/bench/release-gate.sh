#!/bin/sh
set -eu

[ "$#" -eq 1 ] || {
	printf 'usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
	exit 2
}
output=$1
case "$output" in ""|/|.|..|-*) exit 2 ;; esac
[ ! -L "$output" ] || {
	printf 'output directory must not be a symbolic link\n' >&2
	exit 2
}

attempts=${AGEFREIGHTER_BENCH_GATE_ATTEMPTS:-2}
case "$attempts" in ''|*[!0-9]*|0) exit 2 ;; esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$output"
attempt=1
while [ "$attempt" -le "$attempts" ]; do
	attempt_output="$output/attempt-$attempt"
	printf 'Running release performance gate attempt %s of %s\n' \
		"$attempt" "$attempts"
	if "$script_dir/release-budget.sh" "$attempt_output"; then
		printf 'Release performance gate passed on attempt %s\n' "$attempt"
		exit 0
	fi
	if [ "$attempt" -lt "$attempts" ]; then
		printf 'Attempt %s failed; confirming on a clean measurement attempt\n' \
			"$attempt" >&2
	fi
	attempt=$((attempt + 1))
done

printf 'Release performance gate failed on %s consecutive attempts\n' \
	"$attempts" >&2
exit 1
