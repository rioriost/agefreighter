#!/bin/sh

set -eu

if [ "$(id -u)" -ne 0 ] || { [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; }; then
	printf 'usage as root: %s [p0|p1|p2|p3] 4.4.48|5.26.30\n' "$0" >&2
	exit 2
fi

if [ "$#" -eq 1 ]; then
	phase=p0
	version=$1
else
	phase=$1
	version=$2
fi
case "$phase" in
	p0|p1|p2|p3) ;;
	*)
		printf 'unsupported phase: %s\n' "$phase" >&2
		exit 2
		;;
esac
export GOPATH=/root/go
export GOMODCACHE=/root/go/pkg/mod
export GOCACHE=/root/.cache/agefreighter-go-build
mkdir -p "$GOMODCACHE" "$GOCACHE"
case "$version" in
	4.4.48)
		container_name=afps-neo4j44
		image='neo4j@sha256:5098db94262985f26a71d4ff573116cf893bce636e879bceb8ec9ba02a5a1553'
		heap_initial='NEO4J_dbms_memory_heap_initial__size=4G'
		heap_max='NEO4J_dbms_memory_heap_max__size=4G'
		page_cache='NEO4J_dbms_memory_pagecache_size=6G'
		read_only='NEO4J_dbms_databases_default__to__read__only=true'
		;;
	5.26.30)
		container_name=afps-neo4j526
		image='neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d'
		heap_initial='NEO4J_server_memory_heap_initial__size=4G'
		heap_max='NEO4J_server_memory_heap_max__size=4G'
		page_cache='NEO4J_server_memory_pagecache_size=6G'
		read_only='NEO4J_server_databases_default__to__read__only=true'
		;;
	*)
		printf 'unsupported Neo4j version: %s\n' "$version" >&2
		exit 2
		;;
esac

case "$phase" in
	p0)
		heap_size=4G
		page_cache_size=6G
		fixture_workers=4
		;;
	p1)
		heap_size=16G
		page_cache_size=96G
		fixture_workers=16
		;;
	p2)
		heap_size=8G
		page_cache_size=40G
		fixture_workers=8
		;;
	p3)
		printf '%s source sizing must be frozen after P2\n' "$phase" >&2
		exit 3
		;;
esac
case "$version" in
	4.4.48)
		heap_initial="NEO4J_dbms_memory_heap_initial__size=$heap_size"
		heap_max="NEO4J_dbms_memory_heap_max__size=$heap_size"
		page_cache="NEO4J_dbms_memory_pagecache_size=$page_cache_size"
		;;
	5.26.30)
		heap_initial="NEO4J_server_memory_heap_initial__size=$heap_size"
		heap_max="NEO4J_server_memory_heap_max__size=$heap_size"
		page_cache="NEO4J_server_memory_pagecache_size=$page_cache_size"
		;;
esac

: "${KEY_VAULT_NAME:?KEY_VAULT_NAME is required}"
: "${PRODUCTION_SIMULATION_GIT_REF:?PRODUCTION_SIMULATION_GIT_REF is required}"

device=$(readlink -f /dev/disk/azure/scsi1/lun0)
test -b "$device"
mount_root=/mnt/afps-data
if ! blkid "$device" >/dev/null 2>&1; then
	mkfs.ext4 -F -L afpsdata "$device" >/dev/null
fi
mkdir -p "$mount_root"
if ! mountpoint -q "$mount_root"; then
	mount "$device" "$mount_root"
fi
uuid=$(blkid -s UUID -o value "$device")
if ! grep -q "UUID=$uuid" /etc/fstab; then
	printf 'UUID=%s %s ext4 defaults,nofail 0 2\n' "$uuid" "$mount_root" >>/etc/fstab
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io git golang-go >/dev/null
systemctl enable --now docker >/dev/null

source_root=/opt/agefreighter-production-simulation
if [ ! -d "$source_root/.git" ]; then
	git clone -q https://github.com/rioriost/agefreighter.git "$source_root"
fi
git -C "$source_root" fetch -q origin "$PRODUCTION_SIMULATION_GIT_REF"
git -C "$source_root" checkout -q --detach "$PRODUCTION_SIMULATION_GIT_REF"

fixture_root="$mount_root/fixture-$phase"
if [ ! -f "$fixture_root/manifest.json" ]; then
	PRODUCTION_SIMULATION_APPROVAL="reviewed-$phase" \
		"$source_root/production-simulation/scripts/generate-fixture.sh" \
		"$phase" "$fixture_root" 64 "$fixture_workers"
else
	cd "$source_root"
	go run ./production-simulation/cmd/fixturegen verify \
		--manifest "$fixture_root/manifest.json"
fi

neo4j_data="$mount_root/neo4j-$version-$phase"
if [ ! -d "$neo4j_data/databases" ]; then
	PRODUCTION_SIMULATION_APPROVAL="reviewed-$phase" CONTAINER_RUNTIME=docker \
		"$source_root/production-simulation/scripts/import-neo4j.sh" \
		"$phase" "$version" "$fixture_root" "$neo4j_data"
fi

token=$(curl -fsS -H Metadata:true \
	'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net' |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
neo4j_password=$(curl -fsS -H "Authorization: Bearer $token" \
	"https://$KEY_VAULT_NAME.vault.azure.net/secrets/neo4j-password?api-version=7.4" |
	python3 -c 'import json,sys; print(json.load(sys.stdin)["value"])')

case "$neo4j_password" in
	''|*/*)
		printf 'Neo4j password must be non-empty and must not contain a slash\n' >&2
		exit 5
		;;
esac

# Bootstrap needs a short writable window to create the reviewed source-key
# indexes. Always replace only this named container so a prior read-only run is
# not accidentally reused for that step; the durable database remains on the
# separately mounted data disk.
docker rm -f "$container_name" >/dev/null 2>&1 || true
docker run -d --name "$container_name" --restart unless-stopped \
	--publish 7687:7687 --publish 7474:7474 \
	--ulimit nofile=65536:65536 \
	--volume "$neo4j_data:/data" \
	--env "NEO4J_AUTH=neo4j/$neo4j_password" \
	--env "$heap_initial" --env "$heap_max" --env "$page_cache" \
	"$image" >/dev/null

attempt=0
until docker exec -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
	"$container_name" cypher-shell 'RETURN 1' >/dev/null 2>&1; do
	attempt=$((attempt + 1))
	if [ "$attempt" -ge 60 ]; then
		printf 'Neo4j did not become ready\n' >&2
		exit 4
	fi
	sleep 5
done

index_file="$mount_root/source-key-indexes.cypher"
{
	for label in Supplier Facility Product PurchaseOrder Shipment Lot Location Carrier Customer; do
		printf 'CREATE RANGE INDEX %s_source_key IF NOT EXISTS FOR (n:%s) ON (n.source_key);\n' \
			"$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')" "$label"
	done
	for relationship in SUPPLIES PRODUCED_AT PLACED_WITH CONTAINS FULFILLS ORIGINATES_AT DESTINED_FOR CARRIED_BY INCLUDED_IN; do
		printf 'CREATE RANGE INDEX %s_source_key IF NOT EXISTS FOR ()-[r:%s]-() ON (r.source_key);\n' \
			"$(printf '%s' "$relationship" | tr '[:upper:]' '[:lower:]')" "$relationship"
	done
	printf 'CALL db.awaitIndexes(3600);\n'
} >"$index_file"

docker exec -i -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
	"$container_name" cypher-shell --format plain <"$index_file" >/dev/null

# Recreate the container with the version-specific database read-only setting
# only after every index is online. A P1+ source must never rely solely on
# operator convention for immutability. Stop cleanly so Neo4j 5.x checkpoints
# the newly built indexes; a forced removal can leave them needing a rebuild,
# which is intentionally prohibited after read-only mode is enabled.
docker stop --time 120 "$container_name" >/dev/null
docker rm "$container_name" >/dev/null
docker run -d --name "$container_name" --restart unless-stopped \
	--publish 7687:7687 --publish 7474:7474 \
	--ulimit nofile=65536:65536 \
	--volume "$neo4j_data:/data" \
	--env "NEO4J_AUTH=neo4j/$neo4j_password" \
	--env "$heap_initial" --env "$heap_max" --env "$page_cache" \
	--env "$read_only" \
	"$image" >/dev/null

attempt=0
until docker exec -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
	"$container_name" cypher-shell 'RETURN 1' >/dev/null 2>&1; do
	attempt=$((attempt + 1))
	if [ "$attempt" -ge 60 ]; then
		printf 'Read-only Neo4j did not become ready\n' >&2
		exit 4
	fi
	sleep 5
done

access=$(docker exec -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
	"$container_name" cypher-shell --format plain \
	'SHOW DATABASE neo4j YIELD access RETURN access;' | tail -n 1 | tr -d '"\r')
if [ "$access" != 'read-only' ]; then
	printf 'Neo4j database access is %s, expected read-only\n' "$access" >&2
	exit 6
fi

summary="$mount_root/source-summary-$phase.txt"
{
	printf 'phase=%s\n' "$phase"
	printf 'neo4j_version=%s\n' "$version"
	printf 'heap_size=%s\n' "$heap_size"
	printf 'page_cache_size=%s\n' "$page_cache_size"
	python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("fixture_root_sha256="+d["rootSha256"]); print("expected_vertices="+str(d["plan"]["vertexTotal"])); print("expected_edges="+str(d["plan"]["edgeTotal"]))' \
		"$fixture_root/manifest.json"
	docker exec -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
		"$container_name" cypher-shell --format plain \
		'MATCH (n) WITH count(n) AS vertices MATCH ()-[r]->() RETURN vertices, count(r) AS edges;'
	printf 'store_bytes=%s\n' "$(du -sb "$neo4j_data" | awk '{print $1}')"
	printf 'indexes:\n'
	docker exec -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
		"$container_name" cypher-shell --format plain \
		'SHOW INDEXES YIELD name, state, type RETURN name, state, type ORDER BY name;'
	printf 'memory_recommendation:\n'
	if [ "$version" = '4.4.48' ]; then
		docker exec "$container_name" neo4j-admin memrec
	else
		docker exec "$container_name" neo4j-admin server memory-recommendation
	fi
	printf 'database_access=%s\n' "$access"
} >"$summary"
cat "$summary"

unset neo4j_password token
