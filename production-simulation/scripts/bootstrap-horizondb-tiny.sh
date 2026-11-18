#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import json
import urllib.request
request = urllib.request.Request(
    "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01",
    headers={"Metadata": "true"},
)
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5) as response:
    compute = json.load(response)
tags = {tag["name"]: tag["value"] for tag in compute["tagsList"]}
if tags.get("application") != "agefreighter" or tags.get("phase") != "tiny" or tags.get("purpose") != "horizondb-qualification":
    raise SystemExit("Dedicated tiny qualification VM required")
PY

root=/var/lib/agefreighter-horizon-tiny
if [ -e "$root/source.env" ]; then
	printf 'Refusing to overwrite an existing tiny source environment.\n' >&2
	exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io git ca-certificates curl openssl python3 >/dev/null
systemctl enable --now docker
install -d -m 0700 "$root"
umask 077
neo4j_password=$(openssl rand -hex 24)
postgres_password=$(openssl rand -hex 24)
printf 'AGEFREIGHTER_NEO4J_PASSWORD=%s\nAGEFREIGHTER_POSTGRES_TEST_DSN=postgresql://postgres:%s@127.0.0.1:5432/source?sslmode=disable\n' \
	"$neo4j_password" "$postgres_password" >"$root/source.env"
printf 'POSTGRES_PASSWORD=%s\nPOSTGRES_DB=source\n' "$postgres_password" >"$root/postgres.env"
printf 'NEO4J_AUTH=neo4j/%s\nNEO4J_dbms_memory_heap_initial__size=256m\nNEO4J_dbms_memory_heap_max__size=1G\nNEO4J_dbms_memory_pagecache_size=512m\n' \
	"$neo4j_password" >"$root/neo4j44.env"
printf 'NEO4J_AUTH=neo4j/%s\nNEO4J_server_memory_heap_initial__size=256m\nNEO4J_server_memory_heap_max__size=1G\nNEO4J_server_memory_pagecache_size=512m\n' \
	"$neo4j_password" >"$root/neo4j526.env"

postgres_image='postgres@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94'
neo4j44_image='neo4j@sha256:5098db94262985f26a71d4ff573116cf893bce636e879bceb8ec9ba02a5a1553'
neo4j526_image='neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d'
for image in "$postgres_image" "$neo4j44_image" "$neo4j526_image" golang:1.27.4; do
	docker pull --quiet "$image"
done
docker run --detach --name ht-postgres --restart unless-stopped \
	--env-file "$root/postgres.env" --publish 127.0.0.1:5432:5432 \
	--mount type=volume,source=ht-postgres,target=/var/lib/postgresql/data "$postgres_image" >/dev/null
docker run --detach --name ht-neo4j44 --restart unless-stopped \
	--env-file "$root/neo4j44.env" --publish 127.0.0.1:7687:7687 \
	--mount type=volume,source=ht-neo4j44,target=/data "$neo4j44_image" >/dev/null
docker run --detach --name ht-neo4j526 --restart unless-stopped \
	--env-file "$root/neo4j526.env" --publish 127.0.0.1:7688:7687 \
	--mount type=volume,source=ht-neo4j526,target=/data "$neo4j526_image" >/dev/null

ready=0
for _ in $(seq 1 90); do
	if docker exec ht-postgres pg_isready -U postgres -d source >/dev/null 2>&1 \
		&& docker exec ht-neo4j44 cypher-shell -u neo4j -p "$neo4j_password" 'RETURN 1' >/dev/null 2>&1 \
		&& docker exec ht-neo4j526 cypher-shell -u neo4j -p "$neo4j_password" 'RETURN 1' >/dev/null 2>&1; then
		ready=1
		break
	fi
	sleep 3
done
if [ "$ready" != 1 ]; then
	printf 'Tiny source readiness failed; inspect the retained containers.\n' >&2
	exit 1
fi
docker image inspect "$postgres_image" "$neo4j44_image" "$neo4j526_image" golang:1.27.4 \
	--format '{{json .RepoDigests}}'
printf 'Tiny source services ready; credentials retained only in root-owned source.env.\n'
