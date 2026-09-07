#!/usr/bin/env bash
# Emit a bounded, secret-free health and evidence summary for the dedicated P1
# Cosmos fixture. This command is safe to retain as Managed Run Command output.
set -euo pipefail

readonly work=/var/lib/agefreighter-cosmos
readonly evidence=$work/evidence

test -s "$evidence/source.json"
test -s "$evidence/load.json"

jq -c '{schemaVersion,preparedAt,source,database,container,partitionKey,rows,vertices,edges,managedIdentity,publicNetworkAccess,loaderSHA256,fixtureSHA256}' \
  "$evidence/source.json"
jq -c '{rows,remoteRows,files}' "$evidence/load.json"
sha256sum "$evidence/source.json" "$evidence/load.json"

disk_used=$(df -P "$work" | awk 'NR == 2 {gsub(/%/, "", $5); print $5}')
swap_used=$(free -b | awk '/^Swap:/ {print $3}')
oom_events=$(journalctl -k -b --no-pager 2>/dev/null | \
  grep -Eic 'out of memory|oom-kill|killed process' || true)
printf 'disk_used_percent=%s\nswap_used_bytes=%s\noom_events=%s\n' \
  "$disk_used" "$swap_used" "$oom_events"

test "$disk_used" -lt 80
test "$swap_used" -eq 0
test "$oom_events" -eq 0
