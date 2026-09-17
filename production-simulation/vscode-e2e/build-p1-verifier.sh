#!/usr/bin/env bash
# Isolated read-only qualification executable; never replaces the loader.
set -euo pipefail
umask 077
profile=${1:-raw-id}
case "$profile" in
  raw-id) canonical_root=bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70; canonical_version=agefreighter-production-simulation-v1 ;;
  gremlin-partition64) canonical_root=8a048faa36fad90404c263d3ce75073d117e5d96a15f8a614a42347cbd7a0ef4; canonical_version=agefreighter-production-simulation-gremlin-partition64-v1 ;;
  *) echo 'Unknown P1 qualification profile' >&2; exit 2 ;;
esac
test "$#" -le 1 || exit 2
repo=$(git rev-parse --show-toplevel)
test -z "$(git status --porcelain)" || { echo 'Commit the reviewed tree first' >&2; exit 1; }
revision=$(git rev-parse HEAD)
artifact_dir=$(mktemp -d "$repo/production-simulation/work/vscode-p1-verifier.XXXXXX")
mkdir "$artifact_dir/source" "$artifact_dir/bin"
git archive "$revision" | tar -xf - -C "$artifact_dir/source"
(
  cd "$artifact_dir/source"
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o "$artifact_dir/bin/p1runnerverify" ./production-simulation/cmd/p1runnerverify
)
archive="p1-verifier-${revision:0:12}-linux-amd64.tar.gz"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$artifact_dir/$archive" -C "$artifact_dir/bin" p1runnerverify
checksum=$(shasum -a 256 "$artifact_dir/$archive" | awk '{print $1}')
bytes=$(wc -c < "$artifact_dir/$archive" | tr -d ' ')
jq -n --arg version "2.4.0-dev.${revision:0:12}" --arg commit "$revision" --arg sha256 "$checksum" --arg archive "$archive" --argjson bytes "$bytes" --arg profile "$profile" --arg root "$canonical_root" --arg canonicalVersion "$canonical_version" \
  '{schemaVersion:1,platform:"linux-amd64",version:$version,commit:$commit,sha256:$sha256,bytes:$bytes,archive:$archive,purpose:"p1-read-only-verifier",fixtureRoot:"f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f",canonicalRoot:$root,canonicalVersion:$canonicalVersion,qualificationProfile:$profile}' > "$artifact_dir/manifest.json"
echo "$artifact_dir/manifest.json"
