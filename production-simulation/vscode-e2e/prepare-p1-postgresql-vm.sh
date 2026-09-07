#!/usr/bin/env bash
# Prepare the dedicated P1 PostgreSQL-on-VM source. Separate admin and reader
# passwords arrive as named protected Managed Run Command environment parameters
# and are never written to evidence or retained in the container configuration.
set -euo pipefail
umask 077

protected_admin_password=${adminPassword:-}
protected_source_password=${sourcePassword:-}
if (( ${#protected_admin_password} < 24 || ${#protected_source_password} < 24 )); then
  echo "named protected adminPassword and sourcePassword values are required" >&2
  exit 2
fi
admin_password=$protected_admin_password
source_password=$protected_source_password
unset adminPassword sourcePassword protected_admin_password protected_source_password
readonly storage_account=af83c6b829acdc4405aa2dfb
readonly storage_container=af-83c6b829-acdc-4405-aa2d-fb2f2d99af9f
readonly tls_sha256=3d065fa67927c0292826dd42443db28726d0debe5cfecda8a7f3037dd933c7f2
readonly postgres_image=postgres:18.1@sha256:1090bc3a8ccfb0b55f78a494d76f8d603434f7e4553543d6e807bc7bd6bbd17f
readonly work=/var/lib/agefreighter-source
readonly container=agefreighter-pg18-source
readonly password_file=$work/postgres-init-password
trap 'rm -f "$password_file"' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl docker.io jq openssl >/dev/null
systemctl enable --now docker >/dev/null
install -d -m 0700 "$work" "$work/evidence" "$work/download"
install -d -m 0755 "$work/csv"

azcopy_archive="$work/azcopy.tar.gz"
curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 \
  --max-time 300 https://aka.ms/downloadazcopy-v10-linux -o "$azcopy_archive"
tar -xzf "$azcopy_archive" -C "$work"
azcopy_binary=$(find "$work" -maxdepth 2 -type f -name azcopy -print -quit)
test -n "$azcopy_binary"
install -m 0755 "$azcopy_binary" /usr/local/bin/azcopy
AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/fixture/portable-manifest.json" \
  "$work/portable-manifest.json" --overwrite=false --log-level=ERROR
AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/fixture/pgvm/postgres-tls.tar.gz" \
  "$work/postgres-tls.tar.gz" --overwrite=false --log-level=ERROR
AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/uploads" \
  "$work/download" --recursive=true --overwrite=false --log-level=ERROR

printf '%s  %s\n' "$tls_sha256" "$work/postgres-tls.tar.gz" | sha256sum --check --status
install -d -m 0700 "$work/tls"
tar -xzf "$work/postgres-tls.tar.gz" -C "$work/tls"
openssl verify -CAfile "$work/tls/ca.crt" "$work/tls/server.crt" >/dev/null
openssl x509 -in "$work/tls/server.crt" -noout -checkhost postgres18.azpgvm.internal >/dev/null
openssl x509 -in "$work/tls/server.crt" -noout -checkip 10.246.1.20 >/dev/null
chown -R 999:999 "$work/tls"
chmod 0600 "$work/tls/server.key"

while IFS=$'\t' read -r table file expected; do
  found=$(find "$work/download" -type f -name "${expected}.csv" -print -quit)
  test -n "$found"
  printf '%s  %s\n' "$expected" "$found" | sha256sum --check --status
  # The source mount is read-only, but the unprivileged PostgreSQL container
  # user still needs to traverse the directory and read each fixture.
  install -m 0444 "$found" "$work/csv/$file"
done < <(jq -r '.tables[] | [.name,.csv,.csvSha256] | @tsv' "$work/portable-manifest.json")

docker pull "$postgres_image" >/dev/null
docker rm -f "$container" >/dev/null 2>&1 || true
# Each managed execution is a fresh fixture attempt. A failed earlier attempt
# can leave a partly initialized cluster, so clear only this dedicated data
# directory after the container has been removed.
test "$work" = /var/lib/agefreighter-source
install -d -m 0700 "$work/pgdata"
find "$work/pgdata" -mindepth 1 -delete
chown 999:999 "$work/pgdata"
install -o 999 -g 999 -m 0400 /dev/null "$password_file"
printf '%s' "$admin_password" > "$password_file"
docker run -d --name "$container" --restart no \
  --publish 10.246.1.20:5432:5432 \
  --volume "$work/csv:/csv:ro" --volume "$work/tls:/tls:ro" \
  --volume "$password_file:/run/secrets/postgres-password:ro" \
  --volume "$work/pgdata:/var/lib/postgresql" \
  --env POSTGRES_PASSWORD_FILE=/run/secrets/postgres-password \
  "$postgres_image" \
  -c listen_addresses='*' -c ssl=on -c ssl_cert_file=/tls/server.crt \
  -c ssl_key_file=/tls/server.key -c password_encryption=scram-sha-256 >/dev/null

for _ in $(seq 1 120); do
  if docker exec -e PGPASSWORD="$admin_password" "$container" pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
rm -f "$password_file"
docker exec -e PGPASSWORD="$admin_password" "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres \
  -c 'CREATE DATABASE p1source' >/dev/null
docker exec -i -e PGPASSWORD="$admin_password" "$container" psql -v ON_ERROR_STOP=1 -U postgres -d p1source \
  -v reader_password="$source_password" <<'SQL'
CREATE SCHEMA p1;
CREATE ROLE agefreighter_reader LOGIN PASSWORD :'reader_password';
ALTER ROLE agefreighter_reader SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE p1source TO agefreighter_reader;
GRANT USAGE ON SCHEMA p1 TO agefreighter_reader;
SQL

nodes=(Supplier Facility Product PurchaseOrder Shipment Lot Location Carrier Customer)
edges=(SUPPLIES PRODUCED_AT PLACED_WITH CONTAINS FULFILLS ORIGINATES_AT DESTINED_FOR CARRIED_BY INCLUDED_IN)
for table in "${nodes[@]}"; do
  docker exec -i -e PGPASSWORD="$admin_password" "$container" psql -v ON_ERROR_STOP=1 -U postgres -d p1source <<SQL
CREATE TABLE p1."$table" (
  source_key bigint PRIMARY KEY,
  external_id text NOT NULL UNIQUE,
  name text, region text, created_at text, status text, score double precision,
  active boolean, tags jsonb, quantities jsonb, description text
);
COPY p1."$table" FROM '/csv/$table.csv' WITH (FORMAT csv, HEADER true, NULL '\\N');
ANALYZE p1."$table";
GRANT SELECT ON p1."$table" TO agefreighter_reader;
SQL
done
for table in "${edges[@]}"; do
  docker exec -i -e PGPASSWORD="$admin_password" "$container" psql -v ON_ERROR_STOP=1 -U postgres -d p1source <<SQL
CREATE TABLE p1."$table" (
  source_key bigint PRIMARY KEY,
  relationship_id text NOT NULL UNIQUE,
  start_id text NOT NULL, end_id text NOT NULL,
  occurred_at text, quantity bigint, status text, distance_km double precision, notes text
);
COPY p1."$table" FROM '/csv/$table.csv' WITH (FORMAT csv, HEADER true, NULL '\\N');
ANALYZE p1."$table";
GRANT SELECT ON p1."$table" TO agefreighter_reader;
SQL
done

: > "$work/evidence/counts.tsv"
for table in "${nodes[@]}" "${edges[@]}"; do
  count=$(docker exec -e PGPASSWORD="$admin_password" "$container" psql \
    -v ON_ERROR_STOP=1 -U postgres -d p1source -At \
    -c "SELECT count(*) FROM p1.\"$table\"")
  printf '%s\t%s\n' "$table" "$count" >> "$work/evidence/counts.tsv"
done
total=$(awk -F '\t' '{n += $2} END {print n + 0}' "$work/evidence/counts.tsv")
test "$total" -eq 5600000

# The initialization container's bind mount pins the password-file inode even
# after the host pathname is unlinked. Replace that container with an equivalent
# runtime container backed by the completed data directory and no password
# environment variable or secret mount.
docker rm -f "$container" >/dev/null
docker run -d --name "$container" --restart no \
  --publish 10.246.1.20:5432:5432 \
  --volume "$work/csv:/csv:ro" --volume "$work/tls:/tls:ro" \
  --volume "$work/pgdata:/var/lib/postgresql" \
  "$postgres_image" \
  -c listen_addresses='*' -c ssl=on -c ssl_cert_file=/tls/server.crt \
  -c ssl_key_file=/tls/server.key -c password_encryption=scram-sha-256 >/dev/null
for _ in $(seq 1 120); do
  if docker exec "$container" pg_isready -U agefreighter_reader -d p1source >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
docker inspect "$container" | jq '.[0] | {
  image: .Config.Image,
  passwordEnvironment: [.Config.Env[] | select(startswith("POSTGRES_PASSWORD"))],
  secretMounts: [.Mounts[] | select(.Destination == "/run/secrets/postgres-password") | .Destination]
}' > "$work/evidence/container-security.json"
jq -e --arg image "$postgres_image" '.image == $image and .passwordEnvironment == [] and .secretMounts == []' "$work/evidence/container-security.json" >/dev/null
test ! -e "$password_file"

docker exec \
  -e PGPASSWORD="$source_password" -e PGSSLMODE=verify-full -e PGSSLROOTCERT=/tls/ca.crt \
  "$container" psql -h postgres18.azpgvm.internal -U agefreighter_reader -d p1source -At \
  -c 'SELECT current_setting('"'"'transaction_read_only'"'"'), current_setting('"'"'server_version'"'"'), count(*) FROM p1."Supplier"' \
  > "$work/evidence/tls-readonly.txt"

image_digest=$(docker inspect --format '{{index .RepoDigests 0}}' "$postgres_image")
jq -n --arg preparedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg image "$image_digest" --arg tlsSHA256 "$tls_sha256" --argjson rows "$total" \
  '{schemaVersion:1,preparedAt:$preparedAt,source:"postgresql-vm",postgresqlImage:$image,tlsArchiveSHA256:$tlsSHA256,rows:$rows,vertices:1600000,edges:4000000,readOnlyRole:true,publicIP:false}' \
  > "$work/evidence/source.json"
chmod -R go-rwx "$work"
unset admin_password source_password
echo "P1 PostgreSQL VM source prepared and verified"
