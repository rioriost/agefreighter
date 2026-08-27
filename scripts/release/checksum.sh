#!/bin/sh
set -eu

[ "$#" -ge 2 ] || {
	printf 'usage: %s OUTPUT_FILE ARTIFACT...\n' "$0" >&2
	exit 2
}
output=$1
shift
case "$output" in ""|-*|/|.|..|/tmp|/tmp/*|/var/tmp|/var/tmp/*) exit 2 ;; esac
[ ! -L "$output" ] || {
	printf 'checksum output must not be a symbolic link\n' >&2
	exit 2
}

if command -v sha256sum >/dev/null 2>&1; then
	hash() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
	hash() { shasum -a 256 "$1" | awk '{print $1}'; }
else
	printf 'sha256sum or shasum is required\n' >&2
	exit 1
fi

work="$output.unsorted"
rm -f "$work"
trap 'rm -f "$work"' EXIT HUP INT TERM
for artifact do
	case "$artifact" in ""|-*) exit 2 ;; esac
	[ -f "$artifact" ] && [ ! -L "$artifact" ] || {
		printf 'artifact is not a regular file: %s\n' "$artifact" >&2
		exit 1
	}
	base=${artifact##*/}
	printf '%s\n' "$base" | grep -Eq '^[A-Za-z0-9._-]+$' || {
		printf 'unsafe artifact name: %s\n' "$base" >&2
		exit 2
	}
	printf '%s  %s\n' "$(hash "$artifact")" "$base" >>"$work"
done
LC_ALL=C sort -k2,2 "$work" >"$output"
