#!/usr/bin/env bash
# Populate the dedicated P1 Cosmos DB for NoSQL source from the verified
# portable fixture. Authentication is exclusively the VM managed identity.
set -euo pipefail
umask 077

readonly storage_account=af83c6b829acdc4405aa2dfb
readonly storage_container=af-83c6b829-acdc-4405-aa2d-fb2f2d99af9f
readonly endpoint=https://afcosmosp120260907.documents.azure.com:443/
readonly loader_sha256=a891825f120ce9d9c9e9f93ad760d91084ffc0941943c428d7db50def814b6d1
readonly fixture_sha256=1d42cecbac14d98e77151f06030fa8c7fa29d1802b8aae3bdf4c42caa8793eb7
readonly work=/var/lib/agefreighter-cosmos

install -d -m 0700 "$work" "$work/documents" "$work/evidence"
if ! command -v azcopy >/dev/null 2>&1; then
  archive="$work/azcopy.tar.gz"
  curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 \
    --max-time 300 https://aka.ms/downloadazcopy-v10-linux -o "$archive"
  tar -xzf "$archive" -C "$work"
  binary=$(find "$work" -maxdepth 2 -type f -name azcopy -print -quit)
  test -n "$binary"
  install -m 0755 "$binary" /usr/local/bin/azcopy
fi

AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/fixture/portable-manifest.json" \
  "$work/portable-manifest.json" --overwrite=true --log-level=ERROR
AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/fixture/cosmos/cosmosfixtureload-r3" \
  "$work/cosmosfixtureload" --overwrite=true --log-level=ERROR
AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/fixture/cosmos/p1-cosmos-jsonl-clean.tar.gz" \
  "$work/p1-cosmos-jsonl.tar.gz" --overwrite=true --log-level=ERROR

printf '%s  %s\n' "$loader_sha256" "$work/cosmosfixtureload" | sha256sum --check --status
printf '%s  %s\n' "$fixture_sha256" "$work/p1-cosmos-jsonl.tar.gz" | sha256sum --check --status
chmod 0500 "$work/cosmosfixtureload"
test "$work" = /var/lib/agefreighter-cosmos
find "$work/documents" -mindepth 1 -delete
tar -xzf "$work/p1-cosmos-jsonl.tar.gz" -C "$work/documents"

while IFS=$'\t' read -r file expected; do
  printf '%s  %s\n' "$expected" "$work/documents/$file" | sha256sum --check --status
done < <(jq -r '.tables[] | [.documents,.documentsSha256] | @tsv' "$work/portable-manifest.json")

"$work/cosmosfixtureload" \
  -endpoint "$endpoint" -database p1 -container graph \
  -input "$work/documents" -workers 96 \
  > "$work/evidence/load.json" 2> "$work/evidence/load-progress.log"

jq -e --slurpfile manifest "$work/portable-manifest.json" '
  . as $result |
  $result.rows == 5600000 and $result.remoteRows == 5600000 and
  all($manifest[0].tables[]; . as $table | $result.files[$table.documents] == $table.rows)
' "$work/evidence/load.json" >/dev/null
jq -n --arg preparedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg endpoint "$endpoint" --arg loaderSHA256 "$loader_sha256" \
  --arg fixtureSHA256 "$fixture_sha256" \
  '{schemaVersion:1,preparedAt:$preparedAt,source:"cosmos-nosql",endpoint:$endpoint,database:"p1",container:"graph",partitionKey:"/partitionKey",rows:5600000,vertices:1600000,edges:4000000,managedIdentity:true,publicNetworkAccess:false,loaderSHA256:$loaderSHA256,fixtureSHA256:$fixtureSHA256}' \
  > "$work/evidence/source.json"
chmod -R go-rwx "$work/evidence"
echo "P1 Cosmos DB source prepared and verified"
