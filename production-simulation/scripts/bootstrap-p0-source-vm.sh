#!/bin/sh

set -eu

if [ "$(id -u)" -ne 0 ] || [ "$#" -ne 1 ]; then
	printf 'usage as root: %s 4.4.48|5.26.30\n' "$0" >&2
	exit 2
fi

version=$1
export HOME=/root
export GOPATH=/root/go
export GOMODCACHE=/root/go/pkg/mod
mkdir -p "$GOMODCACHE"
case "$version" in
	4.4.48)
		container_name=afps-neo4j44
		image='neo4j@sha256:5098db94262985f26a71d4ff573116cf893bce636e879bceb8ec9ba02a5a1553'
		heap_initial='NEO4J_dbms_memory_heap_initial__size=4G'
		heap_max='NEO4J_dbms_memory_heap_max__size=4G'
		page_cache='NEO4J_dbms_memory_pagecache_size=6G'
		;;
	5.26.30)
		container_name=afps-neo4j526
		image='neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d'
		heap_initial='NEO4J_server_memory_heap_initial__size=4G'
		heap_max='NEO4J_server_memory_heap_max__size=4G'
		page_cache='NEO4J_server_memory_pagecache_size=6G'
		;;
	*)
		printf 'unsupported Neo4j version: %s\n' "$version" >&2
		exit 2
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

fixture_root="$mount_root/fixture-p0"
if [ ! -f "$fixture_root/manifest.json" ]; then
	PRODUCTION_SIMULATION_APPROVAL=reviewed-p0 \
		"$source_root/production-simulation/scripts/generate-fixture.sh" \
		p0 "$fixture_root" 64 4
else
	cd "$source_root"
	go run ./production-simulation/cmd/fixturegen verify \
		--manifest "$fixture_root/manifest.json"
fi

neo4j_data="$mount_root/neo4j-$version"
if [ ! -d "$neo4j_data/databases" ]; then
	PRODUCTION_SIMULATION_APPROVAL=reviewed-p0 CONTAINER_RUNTIME=docker \
		"$source_root/production-simulation/scripts/import-neo4j.sh" \
		p0 "$version" "$fixture_root" "$neo4j_data"
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

if docker container inspect "$container_name" >/dev/null 2>&1; then
	docker start "$container_name" >/dev/null
else
	docker run -d --name "$container_name" --restart unless-stopped \
		--publish 7687:7687 --publish 7474:7474 \
		--ulimit nofile=65536:65536 \
		--volume "$neo4j_data:/data" \
		--env "NEO4J_AUTH=neo4j/$neo4j_password" \
		--env "$heap_initial" --env "$heap_max" --env "$page_cache" \
		"$image" >/dev/null
fi

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

summary="$mount_root/source-summary.txt"
{
	printf 'neo4j_version=%s\n' "$version"
	python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("fixture_root_sha256="+d["rootSha256"]); print("expected_vertices="+str(d["plan"]["vertexTotal"])); print("expected_edges="+str(d["plan"]["edgeTotal"]))' \
		"$fixture_root/manifest.json"
	docker exec -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD="$neo4j_password" \
		"$container_name" cypher-shell --format plain \
		'MATCH (n) WITH count(n) AS vertices MATCH ()-[r]->() RETURN vertices, count(r) AS edges;'
} >"$summary"
cat "$summary"

unset neo4j_password token
