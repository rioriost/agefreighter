#!/bin/bash
set -euo pipefail
umask 077

[[ "$(hostname -s)" == afhs-20261118-p1-csv && "$(id -u)" == 0 ]]
[[ "$(pwd -P)" =~ ^/var/lib/agefreighter-p1/r[0-9]{8}p1postgres(17|18)$ ]]
[[ "$(wc -l </proc/swaps)" -eq 1 ]]
[[ "$(findmnt -no FSTYPE --target /dev/shm)" == tmpfs ]]
secrets=/dev/shm/agefreighter-p1-postgres
container=afh-p1-postgres
image='postgres@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94'
test ! -e postgres-source-runtime.json
mkdir -m 0700 "$secrets"

export DEBIAN_FRONTEND=noninteractive
timeout --kill-after=10s 3m apt-get -o Acquire::Retries=2 update -qq
timeout --kill-after=10s 5m apt-get -o Acquire::Retries=2 install -y -qq docker.io
systemctl enable --now docker
test -z "$(ss -H -ltn 'sport = :5432')"
timeout --kill-after=10s 3m docker pull --quiet "$image"

admin=$(openssl rand -hex 32)
reader=$(openssl rand -hex 32)
printf '%s\n' "$admin" >"$secrets/password"
printf 'postgresql://postgres:%s@127.0.0.1:5432/source?sslmode=disable' "$admin" >"$secrets/admin.dsn"
printf 'postgresql://p1_reader:%s@127.0.0.1:5432/source?sslmode=disable' "$reader" >"$secrets/reader.dsn"
unset admin
docker run --detach --name "$container" --network host --shm-size 256m --memory 2g --memory-swap 2g \
  --env POSTGRES_DB=source --env POSTGRES_PASSWORD_FILE=/run/fixture-password \
  --mount "type=bind,source=$secrets/password,target=/run/fixture-password,readonly" \
  --mount type=volume,source=afh-p1-postgres-data,target=/var/lib/postgresql/data \
  "$image" -c listen_addresses=127.0.0.1 -c shared_buffers=256MB >/dev/null
ready=0
for _ in $(seq 1 60); do
  if docker exec "$container" pg_isready -h 127.0.0.1 -U postgres -d source >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]]
[[ "$(ss -H -ltn 'sport = :5432' | awk '{print $4}')" == 127.0.0.1:5432 ]]
rm "$secrets/password"

docker exec "$container" psql -X -U postgres -d source -q -v ON_ERROR_STOP=1 \
  -c 'CREATE ROLE p1_reader LOGIN' \
  -c 'REVOKE ALL ON DATABASE source FROM PUBLIC' \
  -c 'GRANT CONNECT ON DATABASE source TO p1_reader' \
  -c 'ALTER ROLE p1_reader SET default_transaction_read_only = on'
printf '%s\n%s\n' "$reader" "$reader" | docker exec -i "$container" \
  psql -X -U postgres -d source -q -v ON_ERROR_STOP=1 -c '\password p1_reader'
unset reader
version=$(docker exec "$container" psql -XAt -U postgres -d source -c 'SHOW server_version_num')
[[ "$version" == 170006 ]]
[[ "$(docker inspect --format '{{.Config.Image}}' "$container")" == "$image" ]]
printf '{"image":"%s","serverVersionNum":%s,"bind":"127.0.0.1:5432"}\n' "$image" "$version" >postgres-source-runtime.json
