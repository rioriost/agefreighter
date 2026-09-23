#!/usr/bin/env bash
# Exact fixture only. Never print password verifiers, connection strings or logs.
set -euo pipefail
readonly container=agefreighter-pg18-source
test "$(df -P /var/lib/agefreighter-source | awk 'NR==2 {gsub(/%/, "", $5);print $5}')" -lt 80
test "$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)" -eq 0
test "$(docker inspect --format '{{.Config.Image}}' "$container")" = 'postgres:18.1@sha256:1090bc3a8ccfb0b55f78a494d76f8d603434f7e4553543d6e807bc7bd6bbd17f'
if test "$(docker inspect --format '{{.State.Running}}' "$container")" != true; then
  docker start "$container" >/dev/null
fi
for n in $(seq 1 20); do
  docker exec "$container" pg_isready -U postgres -d p1source >/dev/null 2>&1 && break
  sleep 1
done
docker exec -i --user postgres "$container" psql -X -v ON_ERROR_STOP=1 -U postgres -d p1source -At <<'SQL'
BEGIN READ ONLY;
SET LOCAL statement_timeout='10s';
SELECT json_build_object('role',rolname,'login',rolcanlogin,'superuser',rolsuper,'createRole',rolcreaterole,'createDB',rolcreatedb,'replication',rolreplication,'bypassRLS',rolbypassrls,'connectionLimit',rolconnlimit,'validUntil',rolvaliduntil,'expired',coalesce(rolvaliduntil<now(),false),'hasPassword',rolpassword IS NOT NULL,'scram',coalesce(rolpassword LIKE 'SCRAM-SHA-256$%',false),'defaultReadOnly',EXISTS(SELECT 1 FROM pg_db_role_setting WHERE setrole=pg_authid.oid AND setdatabase=0 AND 'default_transaction_read_only=on'=ANY(setconfig))) FROM pg_authid WHERE rolname='agefreighter_reader';
SELECT json_build_object('databaseConnect',has_database_privilege('agefreighter_reader','p1source','CONNECT'),'schemaUsage',has_schema_privilege('agefreighter_reader','p1','USAGE'),'tables',count(*),'selectable',count(*) FILTER(WHERE has_table_privilege('agefreighter_reader',c.oid,'SELECT'))) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='p1' AND c.relkind='r';
SELECT json_build_object('hbaRuleCount',count(*),'hbaParseErrors',count(*) FILTER(WHERE error IS NOT NULL),'scramHostRules',count(*) FILTER(WHERE type LIKE 'host%' AND auth_method='scram-sha-256')) FROM pg_hba_file_rules;
SELECT json_build_object('memberOf',r.rolname) FROM pg_auth_members m JOIN pg_roles r ON r.oid=m.roleid JOIN pg_roles u ON u.oid=m.member WHERE u.rolname='agefreighter_reader';
COMMIT;
SQL
printf 'source_health\n'
df -P /var/lib/agefreighter-source
free -b
journalctl -k -b --no-pager | grep -Eic 'out of memory|oom-kill|killed process' || true
