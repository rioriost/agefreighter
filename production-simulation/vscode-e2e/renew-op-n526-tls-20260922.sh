#!/usr/bin/env bash
# Approved private fixture renewal only; never run against the original source.
# Retain private keys, database, native credentials, container settings and evidence.
set -euo pipefail
umask 077
readonly root=/var/lib/af-op-n526
readonly evidence="$root/evidence/tls-renewal-20260922"
readonly container=af-op-n526-neo4j526
readonly ip=10.246.5.5
readonly image=neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d
readonly expected_ca=6da7aebf5484f43715daa04330fb6d891833e9d22c4f0c5ed571ab53982cc1ca
# The main task sets the fixed approved window before this script is dispatched.
deadline=${1:?absolute approved UTC deadline required}
test "$(id -u)" = 0
test "$(hostname)" = af-op-n526-source
hostname -I | tr ' ' '\n' | grep -Fx "$ip" >/dev/null
now=$(date -u +%s)
end=$(date -u -d "$deadline" +%s)
test "$now" -lt "$end"
test "$((end - now))" -le 7200
test "$(df --output=pcent / | tail -1 | tr -dc '0-9')" -lt 80
test "$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)" = 0
test ! -e "$evidence"
test -d /var/lib/neo4j526-data/databases/neo4j
test "$(docker inspect --format '{{.Config.Image}}' "$container")" = "$image"
test "$(docker inspect --format '{{.State.OOMKilled}}' "$container")" = false
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
  grep -Fx 'NEO4J_dbms_databases_default__to__read__only=true' >/dev/null
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
  grep -Fx 'NEO4J_server_bolt_tls__level=REQUIRED' >/dev/null
printf '%s  %s\n' "$expected_ca" "$root/evidence/ca.crt" | sha256sum --check --status
test -s "$root/ca.key"
test -s "$root/tls/private.key"
key_files_before=$(sha256sum "$root/ca.key" "$root/tls/private.key")
ca_certificate_key=$(openssl x509 -in "$root/evidence/ca.crt" -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum)
ca_retained_key=$(openssl pkey -in "$root/ca.key" -pubout -outform DER | sha256sum)
test "$ca_certificate_key" = "$ca_retained_key"
install -d -m 0700 "$evidence"
cp -p "$root/evidence/ca.crt" "$evidence/ca-before.crt"
cp -p "$root/tls/public.crt" "$evidence/server-before.crt"
docker inspect --format '{{json .Mounts}}' "$container" > "$evidence/mounts-before.json"
docker inspect --format '{{.Image}} {{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' "$container" > "$evidence/container-before.txt"
# Renew public certificates using the existing keys; keys never leave the guest.
openssl req -x509 -key "$root/ca.key" -sha256 -days 7 \
  -subj '/CN=AGEFreighter OP-N526 P1 Test CA' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' -out "$evidence/ca-after.crt"
openssl req -new -key "$root/tls/private.key" -sha256 \
  -subj '/CN=10.246.5.5' -out "$evidence/server.csr"
printf 'subjectAltName=IP:10.246.5.5\nextendedKeyUsage=serverAuth\nkeyUsage=digitalSignature,keyEncipherment\nbasicConstraints=CA:FALSE\n' > "$evidence/server.ext"
openssl x509 -req -in "$evidence/server.csr" -CA "$evidence/ca-after.crt" \
  -CAkey "$root/ca.key" -CAserial "$evidence/ca.srl" -CAcreateserial \
  -days 7 -sha256 -extfile "$evidence/server.ext" -out "$evidence/server-after.crt"
openssl verify -CAfile "$evidence/ca-after.crt" -verify_ip "$ip" "$evidence/server-after.crt"
openssl x509 -in "$evidence/server-after.crt" -noout -checkend 7200
certificate_key=$(openssl x509 -in "$evidence/server-after.crt" -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum)
retained_key=$(openssl pkey -in "$root/tls/private.key" -pubout -outform DER | sha256sum)
test "$certificate_key" = "$retained_key"
test "$(date -u +%s)" -lt "$end"
install -o 7474 -g 7474 -m 0644 "$evidence/server-after.crt" "$root/tls/public.crt"
# Keep the original CA and all prior receipts at their existing paths.
# The new workflow receives ca-after.crt, not the expired historical CA.
docker restart --time 120 "$container" >/dev/null
for attempt in $(seq 1 45); do
  test "$(date -u +%s)" -lt "$end"
  if openssl s_client -brief -connect "$ip:7687" -verify_ip "$ip" -verify_return_error \
    -CAfile "$evidence/ca-after.crt" </dev/null > "$evidence/tls-handshake.txt" 2>&1; then
    grep -F 'Verification: OK' "$evidence/tls-handshake.txt" >/dev/null && break
  fi
  test "$attempt" -lt 45
  sleep 2
done
docker inspect --format '{{json .Mounts}}' "$container" > "$evidence/mounts-after.json"
cmp "$evidence/mounts-before.json" "$evidence/mounts-after.json"
test "$(sha256sum "$root/ca.key" "$root/tls/private.key")" = "$key_files_before"
test "$(docker inspect --format '{{.State.OOMKilled}}' "$container")" = false
docker inspect --format '{{.Image}} {{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' "$container" > "$evidence/container-after.txt"
sha256sum "$evidence/ca-before.crt" "$evidence/ca-after.crt" "$evidence/server-before.crt" \
  "$evidence/server-after.crt" "$evidence/tls-handshake.txt" "$evidence/container-after.txt" > "$evidence/SHA256SUMS"
openssl x509 -in "$evidence/ca-after.crt" -noout -dates -fingerprint -sha256
printf 'Private fixture TLS renewed; GUI password/inventory checks still required.\n'
