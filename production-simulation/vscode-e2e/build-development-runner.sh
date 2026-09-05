#!/usr/bin/env bash
# Local qualification artifact only. Never publishes a release or builds on a guest.
set -euo pipefail
umask 077
repo=$(git rev-parse --show-toplevel)
if [[ -n $(git status --porcelain) ]]; then
  echo 'Commit and review the working tree before building a pinned qualification artifact.' >&2
  exit 1
fi
revision=$(git rev-parse HEAD)
build_version="2.4.0-dev.${revision:0:12}"
build_date=$(git show -s --format=%cI HEAD)
artifact_dir=$(mktemp -d "$repo/production-simulation/work/vscode-runner-build.XXXXXX")
mkdir "$artifact_dir/source" "$artifact_dir/bin"
git archive "$revision" | tar -xf - -C "$artifact_dir/source"
flags="-X github.com/rioriost/agefreighter/internal/version.Version=$build_version -X github.com/rioriost/agefreighter/internal/version.Commit=$revision -X github.com/rioriost/agefreighter/internal/version.BuildDate=$build_date"
(
  cd "$artifact_dir/source"
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "$flags" -o "$artifact_dir/bin/agefreighter" ./cmd/agefreighter
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "$flags" -o "$artifact_dir/bin/agefreighter-tools" ./cmd/agefreighter-tools
)
archive="agefreighter-${build_version}-linux-amd64.tar.gz"
COPYFILE_DISABLE=1 tar -czf "$artifact_dir/$archive" -C "$artifact_dir/bin" agefreighter agefreighter-tools
checksum=$(shasum -a 256 "$artifact_dir/$archive" | awk '{print $1}')
bytes=$(wc -c < "$artifact_dir/$archive" | tr -d ' ')
jq -n --arg version "$build_version" --arg commit "$revision" --arg sha256 "$checksum" --arg archive "$archive" --argjson bytes "$bytes" \
  '{schemaVersion:1,platform:"linux-amd64",version:$version,commit:$commit,sha256:$sha256,bytes:$bytes,archive:$archive}' > "$artifact_dir/manifest.json"
echo "$artifact_dir/manifest.json"
