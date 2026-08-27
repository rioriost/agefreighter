#!/bin/sh
set -eu

[ "$#" -eq 4 ] || {
	printf 'usage: %s ARCHIVE GOOS GOARCH VERSION\n' "$0" >&2
	exit 2
}
archive=$1
goos=$2
goarch=$3
version=$4
case "$archive" in ""|-*) exit 2 ;; esac
printf '%s\n' "$version" | grep -Eq '^v2\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$' || exit 2
case "$version" in
	*-*)
		printf '%s\n' "${version#*-}" | tr . '\n' |
			awk '/^[0-9]+$/ && length > 1 && substr($0, 1, 1) == "0" { bad=1 } END { exit bad }' ||
			exit 2
		;;
esac
[ -f "$archive" ] && [ ! -L "$archive" ] || exit 1

host_os=$(uname -s)
host_arch=$(uname -m)
case "$host_os:$goos" in Darwin:darwin|Linux:linux) ;; *)
	printf 'refusing to execute %s binaries on %s\n' "$goos" "$host_os" >&2
	exit 1
esac
case "$host_arch:$goarch" in x86_64:amd64|amd64:amd64|arm64:arm64|aarch64:arm64) ;; *)
	printf 'refusing to execute %s binaries on %s\n' "$goarch" "$host_arch" >&2
	exit 1
esac

work=${VERIFY_WORK_DIR:-"$PWD/.release-verify-$$"}
case "$work" in ""|/|.|..|-*) exit 2 ;; esac
[ ! -e "$work" ] || {
	printf 'verification work directory already exists: %s\n' "$work" >&2
	exit 1
}
mkdir -p "$work"
trap 'rm -rf "$work"' EXIT HUP INT TERM

tar -tzf "$archive" | awk '
	/(^|\/)\.\.(\/|$)/ || /^\// { bad=1 }
	END { exit bad }
' || {
	printf 'archive contains an unsafe path\n' >&2
	exit 1
}
tar -tvzf "$archive" | awk 'substr($0, 1, 1) == "l" { bad=1 } END { exit bad }' || {
	printf 'archive contains a symbolic link\n' >&2
	exit 1
}
tar -xzf "$archive" -C "$work"
for binary in agefreighter agefreighter-tools; do
	path=$(find "$work" -type f -name "$binary" -print)
	[ "$(printf '%s\n' "$path" | awk 'NF { count++ } END { print count+0 }')" -eq 1 ] || {
		printf 'archive must contain exactly one %s binary\n' "$binary" >&2
		exit 1
	}
	"$path" version | grep -F "${version#v}" >/dev/null
done
