#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
for script in scripts/release/*.sh; do
	sh -n "$script"
done

work="$root/.release-self-check"
[ ! -L "$work" ] || {
	printf 'refusing symbolic-link work directory\n' >&2
	exit 1
}
rm -rf "$work"
mkdir -p "$work"
trap 'rm -rf "$work"' EXIT HUP INT TERM

case "$(uname -s)" in Darwin) goos=darwin ;; Linux) goos=linux ;; *) exit 0 ;; esac
case "$(uname -m)" in x86_64|amd64) goarch=amd64 ;; arm64|aarch64) goarch=arm64 ;; *) exit 0 ;; esac
commit=$(git rev-parse HEAD)
epoch=$(git show -s --format=%ct HEAD)
if date -u -r "$epoch" '+%Y-%m-%dT%H:%M:%SZ' >/dev/null 2>&1; then
	build_date=$(date -u -r "$epoch" '+%Y-%m-%dT%H:%M:%SZ')
else
	build_date=$(date -u -d "@$epoch" '+%Y-%m-%dT%H:%M:%SZ')
fi
SOURCE_DATE_EPOCH=$epoch scripts/release/build.sh \
	v2.0.0 "$commit" "$build_date" "$goos" "$goarch" "$work"
archive="$work/agefreighter_v2.0.0_${goos}_${goarch}.tar.gz"
scripts/release/verify.sh "$archive" "$goos" "$goarch" v2.0.0
scripts/release/checksum.sh "$work/first.txt" "$archive"
SOURCE_DATE_EPOCH=$epoch scripts/release/build.sh \
	v2.0.0 "$commit" "$build_date" "$goos" "$goarch" "$work" >/dev/null
scripts/release/checksum.sh "$work/second.txt" "$archive"
cmp "$work/first.txt" "$work/second.txt"

for target in darwin_amd64 darwin_arm64 linux_amd64 linux_arm64; do
	target_archive="$work/agefreighter_v2.0.0_${target}.tar.gz"
	[ "$target_archive" = "$archive" ] || cp "$archive" "$target_archive"
done
scripts/release/checksum.sh "$work/checksums.txt" \
	"$work"/agefreighter_v2.0.0_*.tar.gz
scripts/release/generate-formula.sh \
	v2.0.0 example/agefreighter "$work" "$work/agefreighter.rb"
grep -F 'bin.install "agefreighter-tools"' "$work/agefreighter.rb" >/dev/null

for invalid in v1.0.0 v2.01.0 v2.0.0-01; do
	if SOURCE_DATE_EPOCH=$epoch scripts/release/build.sh "$invalid" "$commit" \
		"$build_date" "$goos" "$goarch" "$work/invalid" >/dev/null 2>&1; then
		printf 'build script accepted unsafe version %s\n' "$invalid" >&2
		exit 1
	fi
done
printf 'release script self-check passed\n'
