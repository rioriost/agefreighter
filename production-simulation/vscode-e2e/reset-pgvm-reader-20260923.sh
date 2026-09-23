#!/usr/bin/env bash
# Exact existing reader only. Protected input, never trace or persist plaintext.
set -euo pipefail
umask 077
readonly container=agefreighter-pg18-source
readonly evidence=/var/lib/agefreighter-source/evidence/reader-reset-20260923
test ! -e "$evidence"
test "${#AF_READER_PASSWORD}" -eq 64
[[ "$AF_READER_PASSWORD" =~ ^[a-f0-9]{64}$ ]]
test "$(date -u +%s)" -lt 1790129100 # 2026-09-23 02:05:00 UTC
test "$(df -P /var/lib/agefreighter-source | awk 'NR==2 {gsub(/%/, "", $5);print $5}')" -lt 80
test "$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)" -eq 0
test "$(docker inspect --format '{{.Config.Image}}' "$container")" = 'postgres:18.1@sha256:1090bc3a8ccfb0b55f78a494d76f8d603434f7e4553543d6e807bc7bd6bbd17f'
test "$(docker inspect --format '{{.State.Running}}' "$container")" = true
test "$(docker exec --user postgres "$container" psql -X -v ON_ERROR_STOP=1 -U postgres -d p1source -At -c "SELECT rolcanlogin AND NOT rolsuper AND NOT rolcreaterole AND NOT rolcreatedb AND NOT rolreplication AND NOT rolbypassrls AND (rolvaliduntil IS NULL OR rolvaliduntil>now()) FROM pg_roles WHERE rolname='agefreighter_reader'")" = t
install -d -m 0700 "$evidence"
trap 'unset AF_READER_PASSWORD' EXIT
snapshot() {
  docker exec --user postgres "$container" psql -X -v ON_ERROR_STOP=1 -U postgres -d p1source -At -c "SELECT to_jsonb(r) FROM pg_roles r WHERE rolname='agefreighter_reader'"
}
snapshot >"$evidence/role-before.json"
test "$(wc -l < "$evidence/role-before.json")" -eq 1
# psql encrypts the new password before issuing ALTER ROLE; stdin, not argv.
# No tty is allocated, so its two password prompts consume this private pipe.
if ! printf '%s\n%s\n' "$AF_READER_PASSWORD" "$AF_READER_PASSWORD" |
  timeout 30 docker exec -i --user postgres "$container" psql -X -v ON_ERROR_STOP=1 -U postgres -d p1source -c '\password agefreighter_reader' >"$evidence/rotation.log" 2>/dev/null; then
  echo 'reader_rotation_failed_reconcile_before_retry'; exit 10
fi
snapshot >"$evidence/role-after.json"
cmp -s "$evidence/role-before.json" "$evidence/role-after.json"
if ! {
  printf '%s\n' "$AF_READER_PASSWORD"
  printf '%s\n' "SELECT current_user='agefreighter_reader' AND current_database()='p1source' AND current_setting('transaction_read_only')='on';" "SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid();" "SELECT count(*)=18 AND bool_and(has_table_privilege(current_user,c.oid,'SELECT')) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='p1' AND c.relkind='r';"
} | timeout 30 docker exec -i --user postgres "$container" sh -c 'IFS= read -r PGPASSWORD; export PGPASSWORD; exec psql "host=10.246.1.20 port=5432 dbname=p1source user=agefreighter_reader sslmode=verify-full sslrootcert=/tls/ca.crt connect_timeout=15" -X -v ON_ERROR_STOP=1 -At' >"$evidence/reader-check.txt" 2>/dev/null; then
  echo 'reader_login_failed_after_rotation_reconcile_before_retry'; exit 11
fi
test "$(tr '\n' ' ' < "$evidence/reader-check.txt")" = 't t t '
unset AF_READER_PASSWORD
date -u +%Y-%m-%dT%H:%M:%SZ >"$evidence/completed"
sha256sum "$evidence/role-before.json" "$evidence/role-after.json" "$evidence/rotation.log" "$evidence/reader-check.txt" "$evidence/completed" >"$evidence/SHA256SUMS"
echo 'reader_rotated_tls_login_verified_readonly_and_18_table_access_unchanged'
cat "$evidence/SHA256SUMS"
