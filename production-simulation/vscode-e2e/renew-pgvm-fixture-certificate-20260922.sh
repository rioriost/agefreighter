#!/usr/bin/env bash
# Exact P1 PGVM public-leaf renewal; never replace keys, CA, data or credentials.
# The operator must verify ARM ownership and the approved runtime bound first.
set -euo pipefail
umask 077
readonly root=/var/lib/agefreighter-source
readonly evidence="$root/evidence/tls-renewal-20260922"
readonly container=agefreighter-pg18-source
readonly old=0be89b41afc0cfb0afbb27d47befb831a4e076db5df52203b1a94af218a0fbef
readonly renewed=4b10b4f74d827f22967eab716bac117ec25c19ca56070f6131281e25449746f3
readonly ca=0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68
certificateBase64=${certificateBase64:-${1:-}}
stopEpoch=${stopEpoch:-${2:-}}
test -n "$certificateBase64"
[[ "$stopEpoch" =~ ^[0-9]{10}$ ]]
now=$(date -u +%s)
# Allow completion margin, and reject an accidentally unbounded authorization.
test "$stopEpoch" -gt "$((now + 120))"
test "$stopEpoch" -le "$((now + 7200))"
test ! -e "$evidence"
printf '%s  %s\n' "$old" "$root/tls/server.crt" | sha256sum --check --status
printf '%s  %s\n' "$ca" "$root/tls/ca.crt" | sha256sum --check --status
test -s "$root/evidence/source.json"
test "$(wc -l < "$root/evidence/counts.tsv" | tr -d ' ')" -eq 18
test "$(awk -F '\t' '{n += $2} END {print n + 0}' "$root/evidence/counts.tsv")" -eq 5600000
disk_used=$(df -P "$root" | awk 'NR == 2 {gsub(/%/, "", $5); print $5}')
swap_used=$(free -b | awk '/^Swap:/ {print $3}')
oom_events=$(journalctl -k -b --no-pager | grep -Eic 'out of memory|oom-kill|killed process' || true)
test "$disk_used" -lt 80
test "$swap_used" -eq 0
test "$oom_events" -eq 0
was_running=$(docker inspect --format '{{.State.Running}}' "$container")
case "$was_running" in true|false) ;; *) exit 2 ;; esac
install -d -m 0700 "$evidence"
cp -p "$root/tls/server.crt" "$evidence/server-before.crt"
printf '%s' "$certificateBase64" | base64 --decode > "$evidence/server-after.crt"
printf '%s  %s\n' "$renewed" "$evidence/server-after.crt" | sha256sum --check --status
openssl verify -CAfile "$root/tls/ca.crt" "$evidence/server-after.crt"
openssl x509 -in "$evidence/server-after.crt" -noout -checkhost postgres18.azpgvm.internal
openssl x509 -in "$evidence/server-after.crt" -noout -checkip 10.246.1.20
openssl x509 -in "$evidence/server-after.crt" -noout -checkend 345600
certificate_key=$(openssl x509 -in "$evidence/server-after.crt" -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum)
retained_key=$(openssl pkey -in "$root/tls/server.key" -pubout -outform DER | sha256sum)
test "$certificate_key" = "$retained_key"
test "$(date -u +%s)" -lt "$((stopEpoch - 60))"
install -o 999 -g 999 -m 0600 "$evidence/server-after.crt" "$root/tls/server.crt"
if test "$was_running" = true; then
  docker kill --signal HUP "$container" >/dev/null
else
  docker start "$container" >/dev/null
fi
openssl x509 -in "$root/tls/server.crt" -noout -dates -fingerprint -sha256
sha256sum "$evidence/server-before.crt" "$evidence/server-after.crt"
printf 'disk_used_percent=%s\nswap_used_bytes=%s\noom_events=%s\n' "$disk_used" "$swap_used" "$oom_events"
printf '%s\n' 'Public leaf installed. Private key, CA and database unchanged; live TLS still requires verification.'
