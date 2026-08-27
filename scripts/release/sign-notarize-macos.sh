#!/bin/sh
set -eu

usage() {
	printf 'usage: %s ARCHIVE VERSION ARCH\n' "$0" >&2
	exit 2
}

[ "$#" -eq 3 ] || usage
archive=$1
version=$2
arch=$3

printf '%s\n' "$version" |
	grep -Eq '^v2\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$' ||
	usage
case "$arch" in amd64|arm64) ;; *) usage ;; esac
case "$archive" in ""|/|.|..|-*|*.tmp|/tmp|/tmp/*|/var/tmp|/var/tmp/*) usage ;; esac
[ -f "$archive" ] && [ ! -L "$archive" ] || {
	printf 'archive must be a regular file: %s\n' "$archive" >&2
	exit 1
}
archive=$(CDPATH= cd "$(dirname "$archive")" && pwd)/$(basename "$archive")
[ "$(uname -s)" = Darwin ] || {
	printf 'macOS is required for signing and notarization\n' >&2
	exit 1
}
for command in codesign ditto gzip security tar uuidgen xcrun; do
	command -v "$command" >/dev/null 2>&1 || {
		printf '%s is required\n' "$command" >&2
		exit 1
	}
done

: "${APPLE_DEVELOPER_ID_CERTIFICATE_P12:?missing certificate}"
: "${APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD:?missing certificate password}"
: "${APPLE_SIGNING_IDENTITY:?missing signing identity}"
: "${APPLE_API_KEY_P8:?missing App Store Connect API key}"
: "${APPLE_API_KEY_ID:?missing App Store Connect API key ID}"
: "${APPLE_API_ISSUER_ID:?missing App Store Connect API issuer ID}"
printf '%s\n' "${SOURCE_DATE_EPOCH:-}" | grep -Eq '^[0-9]+$' || {
	printf 'SOURCE_DATE_EPOCH must be a non-negative integer\n' >&2
	exit 2
}

temporary_root=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
work=$(mktemp -d "$temporary_root/agefreighter-notary.XXXXXX")
case "$work" in
	"$temporary_root"/agefreighter-notary.*) ;;
	*)
		printf 'unsafe temporary directory: %s\n' "$work" >&2
		exit 1
		;;
esac
keychain="$work/signing.keychain-db"
keychain_password=$(uuidgen)
cleanup() {
	security delete-keychain "$keychain" >/dev/null 2>&1 || true
	rm -rf "$work"
}
trap cleanup EXIT HUP INT TERM

certificate="$work/developer-id.p12"
api_key="$work/AuthKey_${APPLE_API_KEY_ID}.p8"
printf '%s' "$APPLE_DEVELOPER_ID_CERTIFICATE_P12" |
	/usr/bin/base64 -D >"$certificate"
printf '%s' "$APPLE_API_KEY_P8" |
	/usr/bin/base64 -D >"$api_key"
chmod 600 "$certificate" "$api_key"

security create-keychain -p "$keychain_password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$keychain_password" "$keychain"
security import "$certificate" \
	-k "$keychain" \
	-P "$APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD" \
	-T /usr/bin/codesign
security set-key-partition-list \
	-S apple-tool:,apple:,codesign: \
	-s \
	-k "$keychain_password" \
	"$keychain" >/dev/null
security find-identity -v -p codesigning "$keychain" |
	grep -F "\"$APPLE_SIGNING_IDENTITY\"" >/dev/null || {
	printf 'signing identity was not imported: %s\n' "$APPLE_SIGNING_IDENTITY" >&2
	exit 1
}

stage="$work/agefreighter_${version}_darwin_${arch}"
mkdir "$stage"
tar -xzf "$archive" -C "$stage"
for binary in agefreighter agefreighter-tools; do
	path="$stage/$binary"
	[ -f "$path" ] && [ ! -L "$path" ] && [ -x "$path" ] || {
		printf 'archive is missing executable %s\n' "$binary" >&2
		exit 1
	}
done
[ "$(find "$stage" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')" -eq 2 ] &&
	[ -z "$(find "$stage" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ] || {
	printf 'archive contains unexpected entries\n' >&2
	exit 1
}
mtime_reference="$work/mtime-reference"
touch -r "$stage/agefreighter" "$mtime_reference"

for binary in agefreighter agefreighter-tools; do
	path="$stage/$binary"
	codesign \
		--force \
		--keychain "$keychain" \
		--options runtime \
		--sign "$APPLE_SIGNING_IDENTITY" \
		--timestamp \
		"$path"
	codesign --verify --strict --verbose=2 "$path"
done

submission="$work/notarization.zip"
ditto -c -k --keepParent "$stage" "$submission"
result="$work/notarization.json"
xcrun notarytool submit "$submission" \
	--issuer "$APPLE_API_ISSUER_ID" \
	--key "$api_key" \
	--key-id "$APPLE_API_KEY_ID" \
	--output-format json \
	--timeout 20m \
	--wait >"$result"
grep -Eq '"status"[[:space:]]*:[[:space:]]*"Accepted"' "$result" || {
	cat "$result" >&2
	exit 1
}

for binary in agefreighter agefreighter-tools; do
	touch -r "$mtime_reference" "$stage/$binary"
	codesign --verify --strict --verbose=2 "$stage/$binary"
done

replacement="$archive.tmp"
rm -f "$replacement"
COPYFILE_DISABLE=1 tar \
	--format ustar \
	--gid 0 \
	--gname root \
	--uid 0 \
	--uname root \
	-cf - \
	-C "$stage" \
	agefreighter agefreighter-tools |
	gzip -n >"$replacement"
mv "$replacement" "$archive"
