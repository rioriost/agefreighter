#!/bin/sh
set -eu

usage() {
	printf 'usage: %s VERSION COMMIT BUILD_DATE GOOS GOARCH OUTPUT_DIRECTORY\n' "$0" >&2
	exit 2
}

[ "$#" -eq 6 ] || usage
version=$1
commit=$2
build_date=$3
goos=$4
goarch=$5
output=$6

case "$version" in
	v2.[0-9]*.[0-9]*|v2.[0-9]*.[0-9]*-*)
		printf '%s\n' "$version" | grep -Eq '^v2\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$' || usage
		;;
*) usage ;;
esac
case "$version" in
	*-*)
		printf '%s\n' "${version#*-}" | tr . '\n' |
			awk '/^[0-9]+$/ && length > 1 && substr($0, 1, 1) == "0" { bad=1 } END { exit bad }' ||
			usage
		;;
esac
printf '%s\n' "$commit" | grep -Eq '^[0-9a-f]{7,40}$' || usage
printf '%s\n' "$build_date" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' || usage
case "$goos/$goarch" in
	darwin/amd64|darwin/arm64|linux/amd64|linux/arm64|windows/amd64) ;;
	*) usage ;;
esac
case "$output" in
	""|-*|/|.|..) usage ;;
esac
[ ! -L "$output" ] || {
	printf 'output directory must not be a symbolic link: %s\n' "$output" >&2
	exit 2
}
command -v go >/dev/null 2>&1 || {
	printf 'go is required\n' >&2
	exit 1
}
if [ "$goos" = windows ]; then
	command -v zip >/dev/null 2>&1 || {
		printf 'zip is required for Windows archives\n' >&2
		exit 1
	}
else
	command -v tar >/dev/null 2>&1 || {
		printf 'tar is required for Unix archives\n' >&2
		exit 1
	}
	command -v gzip >/dev/null 2>&1 || {
		printf 'gzip is required for Unix archives\n' >&2
		exit 1
	}
fi

epoch=${SOURCE_DATE_EPOCH:-}
printf '%s\n' "$epoch" | grep -Eq '^[0-9]+$' || {
	printf 'SOURCE_DATE_EPOCH must be a non-negative integer\n' >&2
	exit 2
}

mkdir -p "$output"
output=$(CDPATH= cd "$output" && pwd)
case "$output" in /|/tmp|/tmp/*|/var/tmp|/var/tmp/*)
	printf 'refusing unsafe output directory: %s\n' "$output" >&2
	exit 2
esac
name="agefreighter_${version}_${goos}_${goarch}"
work="$output/.work-$goos-$goarch"
stage="$work/$name"
rm -rf "$work"
mkdir -p "$stage"
trap 'rm -rf "$work"' EXIT HUP INT TERM

suffix=
[ "$goos" != windows ] || suffix=.exe
ldflags="-s -w -buildid= -X github.com/rioriost/agefreighter/internal/version.Version=${version#v} -X github.com/rioriost/agefreighter/internal/version.Commit=$commit -X github.com/rioriost/agefreighter/internal/version.BuildDate=$build_date"
for binary in agefreighter agefreighter-tools; do
	CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" GOFLAGS=-mod=readonly \
		go build -trimpath -buildvcs=false -ldflags "$ldflags" \
		-o "$stage/$binary$suffix" "./cmd/$binary"
done

# Normalize mtimes before archiving. GNU tar additionally normalizes archive headers.
touch_time=$(printf '%s' "$build_date" |
	sed 's/^\(....\)-\(..\)-\(..\)T\(..\):\(..\):\(..\)Z$/\1\2\3\4\5.\6/')
TZ=UTC touch -t "$touch_time" "$stage"/* "$stage"
if [ "$goos" = windows ]; then
	archive="$output/$name.zip"
	rm -f "$archive.tmp"
	(
		cd "$stage"
		zip -X -q "$archive.tmp" "agefreighter.exe" "agefreighter-tools.exe"
	)
	mv "$archive.tmp" "$archive"
elif tar --version 2>/dev/null | grep -q 'GNU tar'; then
	archive="$output/$name.tar.gz"
	rm -f "$archive.tmp"
	tar --sort=name --format=ustar --mtime="@$epoch" --owner=0 --group=0 \
		--numeric-owner -cf - -C "$stage" \
		agefreighter agefreighter-tools | gzip -n >"$archive.tmp"
	mv "$archive.tmp" "$archive"
else
	archive="$output/$name.tar.gz"
	rm -f "$archive.tmp"
	tar --format ustar --uid 0 --gid 0 --uname root --gname root \
		-cf - -C "$stage" agefreighter agefreighter-tools |
		gzip -n >"$archive.tmp"
	mv "$archive.tmp" "$archive"
fi

printf '%s\n' "$archive"
