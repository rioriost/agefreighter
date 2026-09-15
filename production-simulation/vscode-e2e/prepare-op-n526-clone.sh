#!/usr/bin/env bash
# Fixture harness only. Never run on the original qualified source.
set -euo pipefail
umask 077
ip=10.246.5.5
root=/var/lib/af-op-n526
tls="$root/tls"
image=neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d
test "$(id -u)" = 0
hostname -I | tr ' ' '\n' | grep -Fx "$ip" >/dev/null
test "$(date -u +%s)" -lt "$(date -u -d 2026-09-16T07:14:35.311Z +%s)"
test "$(df --output=pcent / | tail -1 | tr -dc '0-9')" -lt 80
test "$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)" = 0
test -d /var/lib/neo4j526-data/databases/neo4j
test ! -e "$root"
test "$(docker inspect --format '{{.State.OOMKilled}}' af-n526-neo4j526)" = false
docker image inspect "$image" >/dev/null
install -d -m 700 "$root" "$root/evidence"
chmod 711 "$root"
docker inspect --format '{{.Image}} {{.State.Status}} {{.RestartCount}}' af-n526-neo4j526 > "$root/evidence/copied-container-before.txt"
docker update --restart=no af-n526-neo4j526 >/dev/null
docker stop --time 120 af-n526-neo4j526 >/dev/null
install -d -m 755 "$tls"
openssl req -x509 -newkey rsa:3072 -nodes -sha256 -days 7 \
  -subj '/CN=AGEFreighter OP-N526 P1 Test CA' \
  -keyout "$root/ca.key" -out "$root/evidence/ca.crt" 2>/dev/null
openssl req -new -newkey rsa:3072 -nodes -sha256 \
  -subj '/CN=10.246.5.5' -keyout "$tls/private.key" -out "$root/server.csr" 2>/dev/null
printf 'subjectAltName=IP:10.246.5.5\nextendedKeyUsage=serverAuth\nkeyUsage=digitalSignature,keyEncipherment\nbasicConstraints=CA:FALSE\n' > "$root/server.ext"
openssl x509 -req -in "$root/server.csr" -CA "$root/evidence/ca.crt" \
  -CAkey "$root/ca.key" -CAcreateserial -days 7 -sha256 \
  -extfile "$root/server.ext" -out "$tls/public.crt" 2>/dev/null
openssl verify -CAfile "$root/evidence/ca.crt" -verify_ip "$ip" "$tls/public.crt"
chown -R 7474:7474 "$tls"
chmod 600 "$tls/private.key"
chmod 644 "$tls/public.crt"
docker run -d --name af-op-n526-neo4j526 --restart unless-stopped --publish 7687:7687 \
  --ulimit nofile=65536:65536 --volume /var/lib/neo4j526-data:/data --volume "$tls:/ssl/bolt:ro" \
  --env NEO4J_server_memory_heap_initial__size=4G \
  --env NEO4J_server_memory_heap_max__size=4G \
  --env NEO4J_server_memory_pagecache_size=8G \
  --env NEO4J_dbms_databases_default__to__read__only=true \
  --env NEO4J_server_bolt_tls__level=REQUIRED \
  --env NEO4J_dbms_ssl_policy_bolt_enabled=true \
  --env NEO4J_dbms_ssl_policy_bolt_base__directory=/ssl/bolt \
  --env NEO4J_dbms_ssl_policy_bolt_private__key=private.key \
  --env NEO4J_dbms_ssl_policy_bolt_public__certificate=public.crt \
  --env NEO4J_dbms_ssl_policy_bolt_client__auth=NONE "$image" >/dev/null
for attempt in $(seq 1 60); do
  if openssl s_client -brief -connect "$ip:7687" -verify_ip "$ip" -verify_return_error \
    -CAfile "$root/evidence/ca.crt" </dev/null > "$root/evidence/tls-handshake.txt" 2>&1; then
    grep -F 'Verification: OK' "$root/evidence/tls-handshake.txt" >/dev/null && break
  fi
  test "$attempt" -lt 60
  sleep 2
done
docker inspect --format '{{.Image}} {{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}' af-op-n526-neo4j526 > "$root/evidence/container-after.txt"
sha256sum "$root/evidence/ca.crt" "$root/evidence/tls-handshake.txt" "$root/evidence/container-after.txt" > "$root/evidence/SHA256SUMS"
printf 'OP-N526 clone TLS readiness passed; authenticated GUI assessment still required.\n'
