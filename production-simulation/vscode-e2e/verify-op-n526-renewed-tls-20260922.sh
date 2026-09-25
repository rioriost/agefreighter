#!/usr/bin/env bash
# Read-only reconciliation of the completed renewal; write new evidence only.
# Never reissue a certificate or restart the database in this reconciliation.
set -euo pipefail
umask 077
readonly root=/var/lib/af-op-n526
readonly evidence="$root/evidence/tls-renewal-20260922"
readonly container=af-op-n526-neo4j526
test "$(curl --noproxy '*' --connect-timeout 3 --max-time 5 -fsS -H Metadata:true \
  'http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text')" = af-op-n526-source
test "$(date -u +%s)" -lt "$(date -u -d 2026-09-22T09:15:00Z +%s)"
test ! -e "$evidence/reconciled-SHA256SUMS"
python3 -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); assert sorted(a,key=lambda x:x["Destination"]) == sorted(b,key=lambda x:x["Destination"]), "Mount configuration changed"; print("All mount fields agree; only array order differed")' \
  "$evidence/mounts-before.json" "$evidence/mounts-after.json"
cert_key() { openssl x509 -in "$1" -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum; }
test "$(cert_key "$evidence/ca-before.crt")" = "$(cert_key "$evidence/ca-after.crt")"
test "$(cert_key "$evidence/ca-after.crt")" = "$(openssl pkey -in "$root/ca.key" -pubout -outform DER | sha256sum)"
test "$(cert_key "$evidence/server-before.crt")" = "$(cert_key "$evidence/server-after.crt")"
test "$(cert_key "$evidence/server-after.crt")" = "$(openssl pkey -in "$root/tls/private.key" -pubout -outform DER | sha256sum)"
cmp "$evidence/server-after.crt" "$root/tls/public.crt"
openssl verify -CAfile "$evidence/ca-after.crt" -verify_ip 10.246.5.5 "$evidence/server-after.crt"
timeout 10s openssl s_client -brief -connect 10.246.5.5:7687 -verify_ip 10.246.5.5 -verify_return_error \
  -CAfile "$evidence/ca-after.crt" </dev/null > "$evidence/tls-reconciled.txt" 2>&1
grep -F 'Verification: OK' "$evidence/tls-reconciled.txt"
test "$(docker inspect --format '{{.State.OOMKilled}}' "$container")" = false
test "$(docker inspect --format '{{.State.Status}}' "$container")" = running
docker inspect --format '{{.Image}} {{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' "$container" > "$evidence/container-reconciled.txt"
sha256sum "$evidence/ca-before.crt" "$evidence/ca-after.crt" "$evidence/server-before.crt" \
  "$evidence/server-after.crt" "$evidence/mounts-before.json" "$evidence/mounts-after.json" \
  "$evidence/tls-handshake.txt" "$evidence/tls-reconciled.txt" "$evidence/container-reconciled.txt" > "$evidence/reconciled-SHA256SUMS"
sha256sum --check "$evidence/reconciled-SHA256SUMS"
openssl x509 -in "$evidence/ca-after.crt" -noout -dates -fingerprint -sha256
# This is the public CA only, for the new GUI workflow; no private key export.
cat "$evidence/ca-after.crt"
