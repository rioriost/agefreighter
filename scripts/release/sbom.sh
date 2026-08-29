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
command -v go >/dev/null 2>&1 || {
	printf 'go is required to locate module license metadata\n' >&2
	exit 1
}
command -v jq >/dev/null 2>&1 || {
	printf 'jq is required to validate SPDX license metadata\n' >&2
	exit 1
}
raw="$output.raw"
rm -f "$raw"
trap 'rm -f "$raw"' EXIT HUP INT TERM
SYFT_GOLANG_LOCAL_MOD_CACHE_DIR=$(go env GOMODCACHE) \
	SYFT_GOLANG_SEARCH_LOCAL_MOD_CACHE_LICENSES=true \
	SYFT_GOLANG_SEARCH_REMOTE_LICENSES=false \
	syft "$archive" --enrich golang -o "spdx-json=$raw"
jq '
  (.packages[] |
    select(.name == "github.com/rioriost/agefreighter") |
    .licenseConcluded) = "MIT" |
  (.packages[] |
    select(.name == "stdlib") |
    .licenseConcluded) = "BSD-3-Clause"
' "$raw" >"$output"
[ -s "$output" ] || {
	printf 'syft did not create %s\n' "$output" >&2
	exit 1
}
jq -e '
  [.packages[] |
    select(any(.externalRefs[]?;
      .referenceType == "purl" and
      (.referenceLocator | startswith("pkg:golang/")))) |
    select(.licenseConcluded == "NOASSERTION")]
  | length == 0
' "$output" >/dev/null || {
	printf 'SPDX contains a Go package without a concluded license\n' >&2
	exit 1
}
rm -f "$raw"
trap - EXIT HUP INT TERM
