#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
	printf 'usage: %s COVERAGE_PROFILE MINIMUM_PERCENT\n' "$0" >&2
	exit 2
fi

profile=$1
minimum=$2
actual=$(go tool cover -func="$profile" | awk '/^total:/ {gsub(/%/, "", $3); print $3}')

if [ -z "$actual" ]; then
	printf 'coverage total not found in %s\n' "$profile" >&2
	exit 1
fi

if ! awk -v value="$actual" 'BEGIN {
	exit !(value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value >= 0 && value <= 100)
}'; then
	printf 'invalid coverage value: %s\n' "$actual" >&2
	exit 1
fi

if ! awk -v value="$minimum" 'BEGIN {
	exit !(value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/ && value >= 0 && value <= 100)
}'; then
	printf 'minimum coverage must be a number from 0 to 100: %s\n' "$minimum" >&2
	exit 2
fi

if ! awk -v actual="$actual" -v minimum="$minimum" 'BEGIN { exit !(actual + 0 >= minimum + 0) }'; then
	printf 'coverage %s%% is below required %s%%\n' "$actual" "$minimum" >&2
	exit 1
fi

printf 'coverage %s%% meets required %s%%\n' "$actual" "$minimum"
