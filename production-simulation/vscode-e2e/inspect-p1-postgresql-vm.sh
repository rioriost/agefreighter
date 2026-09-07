#!/usr/bin/env bash
# Emit a bounded, secret-free health and evidence summary for the dedicated P1
# PostgreSQL-on-VM fixture. This command is safe to retain as Managed Run
# Command output.
set -euo pipefail

readonly work=/var/lib/agefreighter-source
readonly evidence=$work/evidence
readonly container=agefreighter-pg18-source

test -s "$evidence/source.json"
test -s "$evidence/counts.tsv"
test -s "$evidence/container-security.json"
test -s "$evidence/tls-readonly.txt"

jq -c '{schemaVersion,preparedAt,source,postgresqlImage,tlsArchiveSHA256,rows,vertices,edges,readOnlyRole,publicIP}' \
  "$evidence/source.json"
jq -c . "$evidence/container-security.json"
printf 'tls_readonly_result=%s\n' "$(tr -d '\r\n' < "$evidence/tls-readonly.txt")"
printf 'mapped_tables=%s\n' "$(wc -l < "$evidence/counts.tsv" | tr -d ' ')"
printf 'mapped_rows=%s\n' "$(awk -F '\t' '{n += $2} END {print n + 0}' "$evidence/counts.tsv")"
sha256sum "$evidence/source.json" "$evidence/counts.tsv" \
  "$evidence/container-security.json" "$evidence/tls-readonly.txt"

docker inspect "$container" | jq -c '.[0] | {
  image: .Config.Image,
  passwordEnvironment: [.Config.Env[] | select(startswith("POSTGRES_PASSWORD"))],
  secretMounts: [.Mounts[] | select(.Destination == "/run/secrets/postgres-password") | .Destination]
}'
test ! -e "$work/postgres-init-password"

disk_used=$(df -P "$work" | awk 'NR == 2 {gsub(/%/, "", $5); print $5}')
swap_used=$(free -b | awk '/^Swap:/ {print $3}')
oom_events=$(journalctl -k -b --no-pager 2>/dev/null | \
  grep -Eic 'out of memory|oom-kill|killed process' || true)
printf 'disk_used_percent=%s\nswap_used_bytes=%s\noom_events=%s\n' \
  "$disk_used" "$swap_used" "$oom_events"

test "$(wc -l < "$evidence/counts.tsv" | tr -d ' ')" -eq 18
test "$(awk -F '\t' '{n += $2} END {print n + 0}' "$evidence/counts.tsv")" -eq 5600000
test "$disk_used" -lt 80
test "$swap_used" -eq 0
test "$oom_events" -eq 0
