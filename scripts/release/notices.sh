#!/bin/sh
set -eu

[ "$#" -eq 1 ] || {
	printf 'usage: %s OUTPUT_FILE\n' "$0" >&2
	exit 2
}
output=$1
case "$output" in ""|/|.|..|-*) exit 2 ;; esac
[ ! -L "$output" ] || {
	printf 'refusing symbolic-link output: %s\n' "$output" >&2
	exit 1
}
command -v go >/dev/null 2>&1 || {
	printf 'go is required\n' >&2
	exit 1
}
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

temporary="$output.tmp"
rm -f "$temporary"
trap 'rm -f "$temporary"' EXIT HUP INT TERM

modules=$(go list -deps -f \
	'{{with .Module}}{{if not .Main}}{{.Path}}|{{.Version}}|{{.Dir}}{{end}}{{end}}' \
	./cmd/... | LC_ALL=C sort -u)
[ -n "$modules" ] || {
	printf 'no runtime dependency modules found\n' >&2
	exit 1
}
go_version=$(go env GOVERSION)
go_license="$root/licenses/GO_LICENSE"
[ -f "$go_license" ] && [ ! -L "$go_license" ] || {
	printf 'the vendored Go toolchain license was not found\n' >&2
	exit 1
}

{
	printf 'THIRD-PARTY SOFTWARE NOTICES AND LICENSES\n'
	printf '=========================================\n\n'
	printf 'This file contains notices and license texts for Go modules linked into\n'
	printf 'the agefreighter and agefreighter-tools release binaries.\n'
	printf 'The agefreighter project license is distributed separately as LICENSE.\n'
	printf '\n------------------------------------------------------------------------\n'
	printf 'Go standard library %s — LICENSE\n' "$go_version"
	printf '%s\n\n' '------------------------------------------------------------------------'
	awk '{ sub(/\r$/, ""); print }' "$go_license"

	printf '%s\n' "$modules" | while IFS='|' read -r module_name module_version module_dir; do
		[ -n "$module_name" ] && [ -n "$module_version" ] && [ -d "$module_dir" ] || {
			printf 'invalid module metadata for %s\n' "$module_name" >&2
			exit 1
		}
		found=0
		for license_file in \
			"$module_dir"/LICENSE* \
			"$module_dir"/COPYING* \
			"$module_dir"/NOTICE*; do
			[ -f "$license_file" ] && [ ! -L "$license_file" ] || continue
			found=1
			printf '\n------------------------------------------------------------------------\n'
			printf '%s %s — %s\n' \
				"$module_name" "$module_version" "$(basename "$license_file")"
			printf '%s\n\n' '------------------------------------------------------------------------'
			awk '{ sub(/\r$/, ""); print }' "$license_file"
		done
		[ "$found" -eq 1 ] || {
			printf 'no license or notice file found for %s %s\n' \
				"$module_name" "$module_version" >&2
			exit 1
		}
	done
} >"$temporary"

mv "$temporary" "$output"
trap - EXIT HUP INT TERM
