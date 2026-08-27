#!/bin/sh
set -eu

[ "$#" -eq 2 ] || {
	printf 'usage: %s ARCHIVE OUTPUT.spdx.json\n' "$0" >&2
	exit 2
}
archive=$1
output=$2
case "$archive" in ""|-*) exit 2 ;; esac
case "$output" in ""|-*|/|.|..|/tmp|/tmp/*|/var/tmp|/var/tmp/*) exit 2 ;; esac
[ -f "$archive" ] && [ ! -L "$archive" ] || {
	printf 'archive is not a regular file: %s\n' "$archive" >&2
	exit 1
}
[ ! -L "$output" ] || {
	printf 'SBOM output must not be a symbolic link\n' >&2
	exit 2
}
command -v syft >/dev/null 2>&1 || {
	printf 'syft is required to generate SBOMs (CI pins its version)\n' >&2
	exit 1
}
syft "$archive" -o "spdx-json=$output"
[ -s "$output" ] || {
	printf 'syft did not create %s\n' "$output" >&2
	exit 1
}
