#!/usr/bin/env bash
# Renew only the expired public certificate; preserve the CA, key, data and evidence.
set -euo pipefail
umask 077
readonly root=/var/lib/agefreighter-source
readonly evidence="$root/evidence/tls-renewal-20260913"
readonly old=2c38d122532ff6523eb69e30c25a5b997603851c5e39f2e1032408ba93712652
readonly renewed=0be89b41afc0cfb0afbb27d47befb831a4e076db5df52203b1a94af218a0fbef
certificateBase64=${certificateBase64:-${1:-}}
test -n "${certificateBase64:-}"
test ! -e "$evidence"
printf '%s  %s\n' "$old" "$root/tls/server.crt" | sha256sum --check --status
was_running=$(docker inspect --format '{{.State.Running}}' agefreighter-pg18-source)
case "$was_running" in true|false) ;; *) exit 2 ;; esac
printf 'source_container_was_running=%s\n' "$was_running"
install -d -m 0700 "$evidence"
cp -p "$root/tls/server.crt" "$evidence/server-before.crt"
printf '%s' "$certificateBase64" | base64 --decode > "$evidence/server-after.crt"
printf '%s  %s\n' "$renewed" "$evidence/server-after.crt" | sha256sum --check --status
openssl verify -CAfile "$root/tls/ca.crt" "$evidence/server-after.crt"
openssl x509 -in "$evidence/server-after.crt" -noout -checkhost postgres18.azpgvm.internal
openssl x509 -in "$evidence/server-after.crt" -noout -checkip 10.246.1.20
openssl x509 -in "$evidence/server-after.crt" -noout -checkend 270000
certificate_key=$(openssl x509 -in "$evidence/server-after.crt" -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum)
retained_key=$(openssl pkey -in "$root/tls/server.key" -pubout -outform DER | sha256sum)
test "$certificate_key" = "$retained_key"
install -o 999 -g 999 -m 0600 "$evidence/server-after.crt" "$root/tls/server.crt"
if test "$was_running" = true; then
  docker kill --signal HUP agefreighter-pg18-source >/dev/null
else
  docker start agefreighter-pg18-source >/dev/null
fi
openssl x509 -in "$root/tls/server.crt" -noout -dates -fingerprint -sha256
sha256sum "$evidence/server-before.crt" "$evidence/server-after.crt"
printf '%s\n' 'Retained private key, CA and database unchanged; PostgreSQL started or reloaded.'
