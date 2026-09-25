#!/usr/bin/env bash
# Populate the dedicated private PostgreSQL Flexible Server source from the
# already verified P1 CSV fixture on the PostgreSQL source VM. Both passwords
# arrive as named protected Managed Run Command environment parameters.
set -euo pipefail
umask 077

admin_password=${adminPassword:-}
reader_password=${sourcePassword:-}
if (( ${#admin_password} < 24 || ${#reader_password} < 24 )); then
  echo "named protected adminPassword and sourcePassword values are required" >&2
  exit 2
fi
unset adminPassword sourcePassword

readonly target_host=afpg-p1-source-20260907.postgres.database.azure.com
readonly target_admin=afsourceadmin
readonly csv_dir=/var/lib/agefreighter-source/csv
readonly evidence_dir=/var/lib/agefreighter-source/evidence/flexible-server
readonly image=postgres:18.1

test -d "$csv_dir"
test "$(find "$csv_dir" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')" -eq 18
install -d -m 0700 "$evidence_dir"
docker pull "$image" >/dev/null

export PGPASSWORD=$admin_password
docker run --rm -i --env PGPASSWORD \
  --volume /etc/ssl/certs/ca-certificates.crt:/certs/ca.pem:ro "$image" psql \
  "host=$target_host port=5432 user=$target_admin dbname=postgres sslmode=verify-full sslrootcert=/certs/ca.pem" \
  -v ON_ERROR_STOP=1 <<'SQL'
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE datname = 'p1source' AND pid <> pg_backend_pid();
DROP DATABASE IF EXISTS p1source;
CREATE DATABASE p1source;
SQL

docker run --rm -i --env PGPASSWORD --volume "$csv_dir:/csv:ro" \
  --volume /etc/ssl/certs/ca-certificates.crt:/certs/ca.pem:ro "$image" psql \
  "host=$target_host port=5432 user=$target_admin dbname=p1source sslmode=verify-full sslrootcert=/certs/ca.pem" \
  -v ON_ERROR_STOP=1 -v reader_password="$reader_password" <<'SQL'
CREATE SCHEMA p1;
DO $body$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agefreighter_reader') THEN
    CREATE ROLE agefreighter_reader LOGIN;
  END IF;
END
$body$;
ALTER ROLE agefreighter_reader PASSWORD :'reader_password';
ALTER ROLE agefreighter_reader SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE p1source TO agefreighter_reader;
GRANT USAGE ON SCHEMA p1 TO agefreighter_reader;
SQL

nodes=(Supplier Facility Product PurchaseOrder Shipment Lot Location Carrier Customer)
edges=(SUPPLIES PRODUCED_AT PLACED_WITH CONTAINS FULFILLS ORIGINATES_AT DESTINED_FOR CARRIED_BY INCLUDED_IN)
for table in "${nodes[@]}"; do
  docker run --rm -i --env PGPASSWORD --volume "$csv_dir:/csv:ro" \
    --volume /etc/ssl/certs/ca-certificates.crt:/certs/ca.pem:ro "$image" psql \
    "host=$target_host port=5432 user=$target_admin dbname=p1source sslmode=verify-full sslrootcert=/certs/ca.pem" \
    -v ON_ERROR_STOP=1 <<SQL
CREATE TABLE p1."$table" (
  source_key bigint PRIMARY KEY,
  external_id text NOT NULL UNIQUE,
  name text, region text, created_at text, status text, score double precision,
  active boolean, tags jsonb, quantities jsonb, description text
);
\copy p1."$table" FROM '/csv/$table.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')
ANALYZE p1."$table";
GRANT SELECT ON p1."$table" TO agefreighter_reader;
SQL
done
for table in "${edges[@]}"; do
  docker run --rm -i --env PGPASSWORD --volume "$csv_dir:/csv:ro" \
    --volume /etc/ssl/certs/ca-certificates.crt:/certs/ca.pem:ro "$image" psql \
    "host=$target_host port=5432 user=$target_admin dbname=p1source sslmode=verify-full sslrootcert=/certs/ca.pem" \
    -v ON_ERROR_STOP=1 <<SQL
CREATE TABLE p1."$table" (
  source_key bigint PRIMARY KEY,
  relationship_id text NOT NULL UNIQUE,
  start_id text NOT NULL, end_id text NOT NULL,
  occurred_at text, quantity bigint, status text, distance_km double precision, notes text
);
\copy p1."$table" FROM '/csv/$table.csv' WITH (FORMAT csv, HEADER true, NULL '\\N')
ANALYZE p1."$table";
GRANT SELECT ON p1."$table" TO agefreighter_reader;
SQL
done

count_query=''
for table in "${nodes[@]}" "${edges[@]}"; do
  if [[ -n "$count_query" ]]; then count_query+=' UNION ALL '; fi
  count_query+="SELECT '$table', count(*)::bigint FROM p1.\"$table\""
done
docker run --rm --env PGPASSWORD \
  --volume /etc/ssl/certs/ca-certificates.crt:/certs/ca.pem:ro "$image" psql \
  "host=$target_host port=5432 user=$target_admin dbname=p1source sslmode=verify-full sslrootcert=/certs/ca.pem" \
  -At -F $'\t' -c "$count_query ORDER BY 1" > "$evidence_dir/counts.tsv"
total=$(awk -F '\t' '{n += $2} END {print n + 0}' "$evidence_dir/counts.tsv")
test "$total" -eq 5600000

export PGPASSWORD=$reader_password
docker run --rm --env PGPASSWORD \
  --volume /etc/ssl/certs/ca-certificates.crt:/certs/ca.pem:ro "$image" psql \
  "host=$target_host port=5432 user=agefreighter_reader dbname=p1source sslmode=verify-full sslrootcert=/certs/ca.pem" \
  -At -F $'\t' -c "SELECT current_setting('transaction_read_only'), current_setting('server_version'), count(*) FROM p1.\"Supplier\"" \
  > "$evidence_dir/tls-readonly.txt"
unset PGPASSWORD admin_password reader_password

jq -n --arg preparedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg host "$target_host" --argjson rows "$total" \
  '{schemaVersion:1,preparedAt:$preparedAt,source:"postgresql-flexible-server",host:$host,rows:$rows,vertices:1600000,edges:4000000,readOnlyRole:true,publicAccess:false,sslMode:"verify-full"}' \
  > "$evidence_dir/source.json"
chmod -R go-rwx "$evidence_dir"
echo "P1 PostgreSQL Flexible Server source prepared and verified"
