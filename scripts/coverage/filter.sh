#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
	printf 'usage: %s INPUT_PROFILE OUTPUT_PROFILE ALLOWLIST\n' "$0" >&2
	exit 2
fi

input=$1
output=$2
allowlist=$3
module=$(go list -m)

awk -v module="$module/" '
	NR == FNR {
		line = $0
		sub(/^[[:space:]]+/, "", line)
		sub(/[[:space:]]+$/, "", line)
		if (line != "" && line !~ /^#/) {
			excluded[line] = 1
		}
		next
	}
	FNR == 1 {
		print
		next
	}
	{
		file = $1
		sub(/:[0-9].*$/, "", file)
		sub("^" module, "", file)
		if (!(file in excluded)) {
			print
		}
	}
' "$allowlist" "$input" >"$output"
