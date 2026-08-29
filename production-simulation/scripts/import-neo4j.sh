#!/bin/sh

set -eu
. "$(dirname -- "$0")/common.sh"

if [ "$#" -ne 4 ]; then
	printf 'usage: %s PHASE VERSION FIXTURE_DIR NEW_NEO4J_DATA_DIR\n' "$0" >&2
	exit 2
fi

phase=$1
version=$2
fixture=$3
data_directory=$4
runtime=${CONTAINER_RUNTIME:-docker}

case "$phase" in
	p0|p1|p2|p3) ;;
	*)
		printf 'unsupported live phase: %s\n' "$phase" >&2
	exit 2
	;;
esac

case "$version" in
	4.4.48)
		image='neo4j@sha256:5098db94262985f26a71d4ff573116cf893bce636e879bceb8ec9ba02a5a1553'
		;;
	5.26.30)
		image='neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d'
		;;
	*)
		printf 'unsupported Neo4j version: %s\n' "$version" >&2
		exit 2
		;;
esac

require_phase_approval "$phase"
require_command "$runtime"
require_command go
require_file "$fixture/manifest.json"

# The official image runs neo4j-admin as its unprivileged Neo4j user. Fixtures
# contain synthetic non-secret data and are mounted read-only, so grant only
# the traversal/read bits needed by that user without changing file contents.
find "$fixture" -type d -exec chmod 0755 {} +
find "$fixture" -type f -exec chmod 0644 {} +

if [ -e "$data_directory" ]; then
	if [ ! -d "$data_directory" ] || [ -n "$(find "$data_directory" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
		printf 'Neo4j data path must be a new or empty directory: %s\n' "$data_directory" >&2
		exit 3
	fi
else
	mkdir -p "$data_directory"
fi

cd "$(repository_root)"
go run ./production-simulation/cmd/fixturegen verify --manifest "$fixture/manifest.json"

set -- run --rm \
	--mount "type=bind,source=$fixture,target=/fixture,readonly" \
	--mount "type=bind,source=$data_directory,target=/data" \
	"$image" neo4j-admin

if [ "$version" = '4.4.48' ]; then
	set -- "$@" import --database=neo4j
else
	set -- "$@" database import full neo4j
fi

for label in Supplier Facility Product PurchaseOrder Shipment Lot Location Carrier Customer; do
	set -- "$@" "--nodes=$label=/fixture/headers/nodes/$label.header.csv,/fixture/nodes/$label/part-.*"
done
for relationship in SUPPLIES PRODUCED_AT PLACED_WITH CONTAINS FULFILLS ORIGINATES_AT DESTINED_FOR CARRIED_BY INCLUDED_IN; do
	set -- "$@" "--relationships=$relationship=/fixture/headers/edges/$relationship.header.csv,/fixture/edges/$relationship/part-.*"
done

set -- "$@" --id-type=integer --bad-tolerance=0 \
	--skip-bad-relationships=false --skip-duplicate-nodes=false

printf 'Starting reviewed Neo4j %s offline import with image %s\n' "$version" "$image" >&2
exec "$runtime" "$@"
