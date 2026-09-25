#!/usr/bin/env bash
# Run the same complete mapped-record inventory used by the guided extension,
# without VS Code or macOS SecretStorage. The PostgreSQL password must arrive as
# a protected Managed Run Command parameter. Cosmos uses only the VM identity.
set -euo pipefail
umask 077

readonly source_kind=${sourceKind:-}
readonly run_id=${runId:-}
readonly requested_source_host=${sourceHost:-}
if [[ $source_kind != postgresql && $source_kind != cosmos-nosql ]]; then
  echo "sourceKind must be postgresql or cosmos-nosql" >&2
  exit 2
fi
if [[ ! $run_id =~ ^[a-z0-9][a-z0-9-]{2,62}$ ]]; then
  echo "runId must be a bounded lowercase identifier" >&2
  exit 2
fi
unset sourceKind runId sourceHost
if [[ $source_kind == cosmos-nosql && -n $requested_source_host ]]; then
  echo "sourceHost is supported only for PostgreSQL" >&2
  exit 2
fi
source_host=${requested_source_host:-postgres18.azpgvm.internal}
if [[ $source_kind == postgresql && $source_host != postgres18.azpgvm.internal && $source_host != 10.246.1.20 && $source_host != afpg-p1-source-20260907.postgres.database.azure.com ]]; then
  echo "sourceHost is outside the dedicated P1 fixture" >&2
  exit 2
fi

readonly storage_account=af83c6b829acdc4405aa2dfb
readonly storage_container=af-83c6b829-acdc-4405-aa2d-fb2f2d99af9f
readonly commit=06c0e9f225f2c81db527f519ebba1a42d3d9c7fd
readonly version=2.4.0-dev.06c0e9f225f2
readonly archive=agefreighter-2.4.0-dev.06c0e9f225f2-linux-amd64.tar.gz
readonly archive_sha256=b68c726ddadbb5b902e390a7d4c20cf9c731e128f9ef1641ed892a28186c4355
readonly root=/var/lib/agefreighter-headless-inventory
readonly run=$root/runs/$run_id
test ! -e "$run"
install -d -m 0700 "$root/runs" "$run" "$run/bin"

AZCOPY_AUTO_LOGIN_TYPE=MSI azcopy copy \
  "https://${storage_account}.blob.core.windows.net/${storage_container}/artifacts/development/${commit}/${archive}" \
  "$run/$archive" --overwrite=false --log-level=ERROR
printf '%s  %s\n' "$archive_sha256" "$run/$archive" | sha256sum --check --status
tar -xzf "$run/$archive" -C "$run/bin"
readonly binary=$run/bin/agefreighter
test -x "$binary"
version_output=$($binary version)
printf '%s\n' "$version_output" | \
  grep -F "agefreighter $version (commit: $commit, built:" >/dev/null

readonly nodes='[
  {"label":"Supplier","rows":"40000"},{"label":"Facility","rows":"20000"},
  {"label":"Product","rows":"200000"},{"label":"PurchaseOrder","rows":"450000"},
  {"label":"Shipment","rows":"350000"},{"label":"Lot","rows":"500000"},
  {"label":"Location","rows":"10000"},{"label":"Carrier","rows":"1000"},
  {"label":"Customer","rows":"29000"}
]'
readonly edges='[
  {"label":"SUPPLIES","rows":"400000","start":"Supplier","end":"Product"},
  {"label":"PRODUCED_AT","rows":"300000","start":"Product","end":"Facility"},
  {"label":"PLACED_WITH","rows":"450000","start":"PurchaseOrder","end":"Supplier"},
  {"label":"CONTAINS","rows":"1000000","start":"PurchaseOrder","end":"Product"},
  {"label":"FULFILLS","rows":"450000","start":"Shipment","end":"PurchaseOrder"},
  {"label":"ORIGINATES_AT","rows":"350000","start":"Shipment","end":"Facility"},
  {"label":"DESTINED_FOR","rows":"350000","start":"Shipment","end":"Location"},
  {"label":"CARRIED_BY","rows":"350000","start":"Shipment","end":"Carrier"},
  {"label":"INCLUDED_IN","rows":"350000","start":"Lot","end":"Shipment"}
]'

jq -n --arg source "$source_kind" --argjson nodes "$nodes" --argjson edges "$edges" '
  def runtime: {
    memoryLimit:"4GiB",batchRows:5000,batchBytes:"16MiB",
    maxSourceConcurrency:1,maxTransformConcurrency:1,maxTargetConnections:8,
    operationTimeout:"10m"
  };
  def target: {
    type:"apache-age",graph:(if $source == "postgresql" then "headless_pg" else "headless_cosmos" end),
    mode:"create",connection:{env:"AGEFREIGHTER_TARGET_DSN"},propertyMode:"replace"
  };
  def vertexProperties: {
    name:"name",region:"region",created_at:"created_at",status:"status",score:"score",
    active:"active",tags:"tags",quantities:"quantities",description:"description"
  };
  def edgeProperties: {
    occurred_at:"occurred_at",quantity:"quantity",status:"status",
    distance_km:"distance_km",notes:"notes"
  };
  def pgVertices: $nodes | map({
    label:.label,
    query:("SELECT \"external_id\", \"name\", \"region\", \"created_at\", \"status\", \"score\", \"active\", \"tags\", \"quantities\", \"description\" FROM \"p1\".\"" + .label + "\" ORDER BY \"external_id\""),
    idField:"external_id",properties:vertexProperties
  });
  def pgEdges: $edges | map({
    label:.label,
    query:("SELECT \"relationship_id\", \"start_id\", \"end_id\", \"occurred_at\", \"quantity\", \"status\", \"distance_km\", \"notes\" FROM \"p1\".\"" + .label + "\" ORDER BY \"relationship_id\""),
    externalIdField:"relationship_id",
    start:{label:.start,field:"start_id"},end:{label:.end,field:"end_id"},
    properties:edgeProperties
  });
  def pointerProperties($properties): $properties | with_entries(.value = ("/" + .value));
  def cosmosVertices: $nodes | map({
    container:"graph",label:.label,
    query:"SELECT * FROM c WHERE c[\"label\"] = @label",
    parameters:[{name:"@label",value:.label}],idField:"/external_id",
    properties:pointerProperties(vertexProperties)
  });
  def cosmosEdges: $edges | map({
    container:"graph",label:.label,
    query:"SELECT * FROM c WHERE c[\"label\"] = @label",
    parameters:[{name:"@label",value:.label}],externalIdField:"/relationship_id",
    start:{label:.start,field:"/start_id"},end:{label:.end,field:"/end_id"},
    properties:pointerProperties(edgeProperties)
  });
  {
    apiVersion:"agefreighter.io/v2",kind:"LoadJob",
    metadata:{name:(if $source == "postgresql" then "headless-pg-inventory" else "headless-cosmos-inventory" end)},
    source:(if $source == "postgresql" then {
      type:"postgresql",namespace:"p1",
      postgresql:{connection:{env:"AGEFREIGHTER_SOURCE_DSN"},readMode:"cursor",fetchRows:5000,vertices:pgVertices,edges:pgEdges}
    } else {
      type:"cosmos-nosql",namespace:"p1",
      cosmos:{endpoint:"https://afcosmosp120260907.documents.azure.com:443/",credential:"default-azure",database:"p1",pageSize:1000,vertices:cosmosVertices,edges:cosmosEdges}
    } end),
    target:target,runtime:runtime,
    errors:{malformedRecord:"fail",missingEndpoint:"error",rejectLimit:0}
  }
' > "$run/job.json"

"$binary" validate --format json "$run/job.json" > "$run/validate.json"
jq -e '.valid == true' "$run/validate.json" >/dev/null

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ $source_kind == postgresql ]]; then
  protected_source_password=${sourcePassword:-}
  if (( ${#protected_source_password} < 24 )); then
    echo "protected sourcePassword is required for PostgreSQL" >&2
    exit 2
  fi
  encoded_password=$(jq -rn --arg value "$protected_source_password" '$value | @uri')
  source_ca=/var/lib/agefreighter-source/tls/ca.crt
  if [[ $source_host == afpg-p1-source-20260907.postgres.database.azure.com ]]; then
    source_ca=/etc/ssl/certs/ca-certificates.crt
  fi
  encoded_ca=$(jq -rn --arg value "$source_ca" '$value | @uri')
  unset sourcePassword protected_source_password
  AGEFREIGHTER_SOURCE_DSN="postgresql://agefreighter_reader:${encoded_password}@${source_host}:5432/p1source?sslmode=verify-full&sslrootcert=${encoded_ca}&connect_timeout=15" \
    "$binary" inventory --format json "$run/job.json" > "$run/inventory.json" 2> "$run/inventory.stderr"
  unset encoded_password
else
  test -z "${sourcePassword:-}"
  "$binary" inventory --format json "$run/job.json" > "$run/inventory.json" 2> "$run/inventory.stderr"
fi
finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

access=private-dns
if [[ $source_kind == cosmos-nosql ]]; then
  access=managed-identity-private-endpoint
elif [[ $source_host == 10.246.1.20 ]]; then
  access=ip-port-only
elif [[ $source_host == afpg-p1-source-20260907.postgres.database.azure.com ]]; then
  access=flexible-server-private-dns
fi

readonly expected='{
  "vertex:Supplier":"40000","vertex:Facility":"20000","vertex:Product":"200000",
  "vertex:PurchaseOrder":"450000","vertex:Shipment":"350000","vertex:Lot":"500000",
  "vertex:Location":"10000","vertex:Carrier":"1000","vertex:Customer":"29000",
  "edge:SUPPLIES":"400000","edge:PRODUCED_AT":"300000","edge:PLACED_WITH":"450000",
  "edge:CONTAINS":"1000000","edge:FULFILLS":"450000","edge:ORIGINATES_AT":"350000",
  "edge:DESTINED_FOR":"350000","edge:CARRIED_BY":"350000","edge:INCLUDED_IN":"350000"
}'
jq -e --argjson expected "$expected" '
  ([.sections[] | select(.title == "Source inventory").fields[] | {key:.name,value:.value}] | from_entries) as $source |
  ([.sections[] | select(.title == "Mapped record counts").fields[] | {key:.name,value:.value}] | from_entries) as $actual |
  .schemaVersion == 1 and .command == "inventory" and .outcome == "pass" and
  .errors == [] and .incompleteChecks == [] and
  $source.vertices == "1600000" and $source.edges == "4000000" and $source.totalRows == "5600000" and
  $actual == $expected
' "$run/inventory.json" >/dev/null

jq -n --arg source "$source_kind" --arg run "$run_id" --arg version "$version" \
  --arg commit "$commit" --arg startedAt "$started_at" --arg finishedAt "$finished_at" \
  --arg access "$access" \
  --arg reportSHA256 "$(sha256sum "$run/inventory.json" | awk '{print $1}')" \
  '{schemaVersion:1,source:$source,access:$access,run:$run,version:$version,commit:$commit,startedAt:$startedAt,finishedAt:$finishedAt,rows:5600000,vertices:1600000,edges:4000000,mappedLabels:18,outcome:"pass",reportSHA256:$reportSHA256}' \
  > "$run/result.json"
chmod -R go-rwx "$run"
jq -c . "$run/result.json"
