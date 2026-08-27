# Load job configuration

agefreighter 2.x load jobs use the versioned `agefreighter.io/v2` `LoadJob`
format. YAML and JSON inputs are accepted. Documents are decoded strictly:
unknown fields, multiple YAML documents, unsupported API versions, and literal
credentials are rejected.

The canonical structural definition is
[`load-job.schema.json`](load-job.schema.json). Cross-field constraints, such as
batch memory limits and edge identity requirements for upserts, are enforced by
the CLI in addition to the schema.

Validate a job without connecting to its source or target:

```sh
agefreighter validate job.yaml
```

Print its normalized static execution plan:

```sh
agefreighter plan job.yaml
```

## Credentials

Credentials may only refer to an environment variable or file. Literal secret
values are not supported.

```yaml
connection:
  env: AGEFREIGHTER_TARGET_DSN
```

```yaml
connection:
  file: /run/secrets/age-dsn
```

## Runtime concurrency

`maxSourceConcurrency` is fixed at `1`. Every connector assigns checkpoints,
malformed-record counts, and resume positions in source order. CSV and Cosmos
also depend on sequential file/page state; Neo4j does not provide one shared
snapshot across mapping queries; PostgreSQL can share an exported snapshot,
but emitting mappings concurrently would require retaining and resequencing an
unbounded later mapping. A five-run 200,000-row CSV-to-AGE comparison measured
median throughput of 203,347 rows/s at 1, 195,547 at 2, and 193,461 at 4; the
higher values only enlarged the record channel because no additional reader
was started. Values above 1 are rejected rather than silently ignored.

`maxTargetConnections` bounds the Apache AGE connection pool and must be at
least 2.

`maxTransformConcurrency` is fixed at `1`. Record parsing, validation,
position assignment, malformed-record accounting, and property encoding form
one ordered connector operation. A 200,000-row CSV-to-AGE profile on the
release benchmark showed this transform path at 3.45% of sampled CPU time,
while target I/O and scheduler waits dominated. An ordered worker pool would
therefore add channels, resequencing buffers, and retained memory without a
material end-to-end gain. Values above 1 are rejected rather than silently
ignored.

## CSV options

CSV files use UTF-8 and default to comma delimiters, double-quote quoting and
escaping, a header row, and an empty null representation. Source defaults can
be overridden per vertex or edge file. Per-file options inherit every omitted
value from the source defaults.

```yaml
source:
  type: csv
  namespace: crm
  csv:
    defaults:
      delimiter: ","
      quote: '"'
      escape: '"'
      header: true
      encoding: utf-8
      nullValue: ""
    vertices:
      - label: Person
        path: people.tsv
        idColumn: person_id
        format:
          delimiter: "\t"
        properties:
          name: full_name
```

Delimiter, quote, and escape values must each contain exactly one Unicode code
point. None can be a line break, and the delimiter cannot equal the quote
character.

At the first read, the CSV source fingerprints the complete ordered manifest:
every mapped file's bytes and mapping semantics. Resume refuses any change to
that manifest, including files before or after the checkpoint. An opened file
descriptor is the immutable snapshot for its active mapping, so atomically
replacing that path does not mix old and new bytes in the active read. Do not
modify a mapped file in place while a job is running; such changes violate the
source contract, fail verification, and cannot be resumed against the changed
manifest.

## PostgreSQL options

PostgreSQL sources use a referenced libpq connection string and execute every
mapping against one exported `REPEATABLE READ`, read-only snapshot. The
snapshot-exporting transaction remains open until the iterator closes. The
reader transaction imports that snapshot before executing its source query, so
vertices and edges observe one point in time even when the source is changing.

```yaml
source:
  type: postgresql
  namespace: crm
  postgresql:
    connection:
      env: AGEFREIGHTER_SOURCE_DSN
    readMode: copy
    fetchRows: 1000
    vertices:
      - label: Person
        query: SELECT person_id, full_name FROM people ORDER BY person_id
        idField: person_id
        properties:
          name: full_name
    edges:
      - label: KNOWS
        query: SELECT relationship_id, from_id, to_id FROM knows ORDER BY relationship_id
        externalIdField: relationship_id
        start:
          label: Person
          field: from_id
        end:
          label: Person
          field: to_id
```

`readMode` defaults to `copy`. It streams
`COPY (SELECT row_to_json(...)) TO STDOUT` without materializing the complete
result. `cursor` uses a server-side cursor and retains at most `fetchRows`
rows. For these modes, restart resume reopens the ordered query and skips the
checkpointed row count; queries must therefore use a unique total ordering and
the source must remain unchanged between attempts.

`keyset` is the durable resume mode for very large sources and constrained
append-only sources. Every mapping must set `keyField`, return a strictly
increasing unique signed 64-bit integer key, accept `$1` as the prior key (null
on the first request), and accept `$2` as the fetch limit. Across restart,
existing rows must not change or disappear and new keys must become visible in
strictly increasing commit order. A transaction that commits a lower key after
a higher key was checkpointed cannot be recovered by keyset pagination.
Restricting keys to signed 64-bit integers avoids database-collation
differences for text and precision loss for high-precision numeric values. For
example:

```yaml
readMode: keyset
fetchRows: 1000
vertices:
  - label: Person
    query: >-
      SELECT person_id, full_name
      FROM people
      WHERE ($1::bigint IS NULL OR person_id > $1)
      ORDER BY person_id
      LIMIT $2
    keyField: person_id
    idField: person_id
```

Queries must be one `SELECT` or `WITH` statement without a semicolon and must
contain an explicit `ORDER BY` for deterministic restart.
`fetchRows` defaults to 1000 and must be from 1 to 100000. PostgreSQL nulls,
booleans, signed 64-bit integers, finite floating-point values, strings,
arrays, and objects map to the corresponding agefreighter property types.
Temporal, UUID, network, and other PostgreSQL values emitted by
`row_to_json` map to strings. Unsupported or overflowing values follow the
configured malformed-record policy.

The source fingerprint includes a credential-free identity of the configured
PostgreSQL hosts, user, database, and startup session parameters in addition to
every mapping. Repointing an environment-variable or file secret at a
different database, schema search path, or startup role invalidates an old
checkpoint instead of skipping rows in an unrelated source.

The exported-snapshot owner is intentionally long-lived. It pins the source
database's xmin horizon until the load closes, so long migrations can delay
vacuum cleanup. Configure `idle_in_transaction_session_timeout` above the
maximum job duration. PgBouncer transaction-pooling mode cannot preserve the
required session and transaction semantics; connect directly or use
session-pooling mode. The 2.0.0 iterator processes configured mappings
sequentially to preserve vertex-before-edge and checkpoint order.

## Neo4j options

Neo4j sources use the official Go driver and stream one read-only Cypher result
at a time. `fetchRows` is the Bolt pull size, defaults to 1000, and must be from
1 to 100000; it does not materialize that many records eagerly.

```yaml
source:
  type: neo4j
  namespace: crm
  neo4j:
    uri: bolt://127.0.0.1:7687
    database: neo4j
    sourceId: crm-primary
    username: neo4j
    password:
      env: AGEFREIGHTER_NEO4J_PASSWORD
    fetchRows: 1000
    multiLabelPolicy: configured
    vertices:
      - label: Person
        query: >-
          MATCH (n:Person)
          WHERE $afterKey IS NULL OR n.source_key > $afterKey
          RETURN n.source_key AS source_key, n.person_id AS person_id,
                 n.name AS name
          ORDER BY source_key
        keyField: source_key
        idField: person_id
        properties:
          name: name
    edges:
      - label: KNOWS
        query: >-
          MATCH (a:Person)-[r:KNOWS]->(b:Person)
          WHERE $afterKey IS NULL OR r.source_key > $afterKey
          RETURN r.source_key AS source_key, r.relationship_id AS relationship_id,
                 a.person_id AS from_id, b.person_id AS to_id
          ORDER BY source_key
        keyField: source_key
        externalIdField: relationship_id
        start:
          label: Person
          field: from_id
        end:
          label: Person
          field: to_id
```

Instead of explicit `vertices` and `edges`, Neo4j can discover mappings before
opening or admitting the target graph:

```yaml
source:
  type: neo4j
  namespace: crm
  neo4j:
    uri: bolt://127.0.0.1:7687
    database: neo4j
    sourceId: crm-primary
    username: neo4j
    password:
      env: AGEFREIGHTER_NEO4J_PASSWORD
    discovery:
      enabled: true
      labelPrefix: App
      relationshipTypePrefix: APP_
      vertexKeyProperty: source_key
      vertexIdProperty: person_id
      edgeKeyProperty: source_key
      edgeIdProperty: relationship_id
      maxLabels: 256
      maxProperties: 1024
```

Discovery and explicit mappings are mutually exclusive. `labelPrefix` and
`relationshipTypePrefix` select source labels and relationship types without
renaming them. With no `labelPrefix`, unlabeled nodes are included under the
target label `NO_LABEL`; a source label named `NO_LABEL` conflicts with that
mapping. Relationships whose endpoints are outside the selected labels are
omitted.

Every selected node must contain `vertexKeyProperty` and `vertexIdProperty`;
every selected relationship must contain `edgeKeyProperty` and
`edgeIdProperty`. Identity properties default to their corresponding key
properties when omitted. Key values must satisfy the same unique, strictly
increasing signed 64-bit integer contract as explicit mappings. Discovery
never uses Neo4j internal IDs or `elementId()`. All discovered properties are
copied.

A node with multiple selected labels is assigned once, to its lexicographically
first selected label. Edge endpoint mappings use the same rule, and discovery
therefore requires `multiLabelPolicy: configured`. Discovery is rerun for load,
resume, and verify; the resolved mappings are fingerprinted, so a schema or
mapping change rejects resume and verification. `maxLabels` bounds both labels
and relationship types, `maxProperties` bounds the properties of each, and
generated mappings are capped at 1024. Discovery queries and subsequent mapping
queries are separate read transactions, so concurrent source mutations can
still produce inconsistent results.

Every mapping must return a unique, strictly increasing signed 64-bit integer
`keyField`, reference the nullable `$afterKey` parameter, use a top-level
ascending `ORDER BY keyField` as the first ordering expression, and avoid
`SKIP`, `OFFSET`, `LIMIT`, `UNION`, and eager `collect` expressions. The
driver's Bolt fetch size bounds full-result streaming instead. Checkpoints
store the last committed key rather
than a reusable Neo4j internal ID. Across restart, existing rows must not
change or disappear and new keys must become visible in strictly increasing
commit order; a late commit below the checkpoint cannot be recovered.
`sourceId` is an explicit stable dataset identity and is fingerprinted with the
URI, database, username, and ordered mappings. Change it when a URI is
repointed at a logically different graph.

`multiLabelPolicy: configured` always uses the mapping's configured target
label and converts returned Neo4j node values from their properties, ignoring
additional source labels. `reject` treats a returned node with more than one
source label as malformed. Relationship values similarly expose only their
properties; Neo4j internal and element IDs are never selected automatically as
durable identities.

Neo4j nulls, booleans, signed integers, finite floats, strings, lists, and maps
map recursively. Date/time values become deterministic strings. Durations
become objects containing `months`, `days`, `seconds`, and `nanoseconds`.
Two- and three-dimensional points become objects containing `srid` and their
coordinates. Unsupported paths, byte arrays, non-finite values, excessive
nesting, and invalid graph values follow the malformed-record policy.

Each configured mapping runs in its own read transaction. A single mapping is
streamed consistently, but multiple vertex and edge mappings do not share a
point-in-time graph snapshot. Mutations committed between mappings can
therefore be observed by later mappings; the static load plan always reports
this consistency limitation.

## Cosmos DB for NoSQL options

Cosmos sources authenticate with `DefaultAzureCredential`; account keys and
connection strings are not accepted. Queries run across logical partitions and
must use named parameters rather than interpolated values. Every mapped field
is an RFC 6901 JSON Pointer.

```yaml
source:
  type: cosmos-nosql
  namespace: crm
  cosmos:
    endpoint: https://example.documents.azure.com:443/
    credential: default-azure
    database: agefreighter
    pageSize: 100
    vertices:
      - container: vertices
        label: Person
        query: SELECT * FROM c WHERE c.kind = @kind
        parameters:
          - name: "@kind"
            value: person
        idField: /id
        properties:
          name: /profile/name
    edges:
      - container: edges
        label: KNOWS
        query: SELECT * FROM c WHERE c.kind = @kind
        parameters:
          - name: "@kind"
            value: knows
        externalIdField: /id
        start:
          label: Person
          field: /from/id
        end:
          label: Person
          field: /to/id
```

`pageSize` defaults to 100 and must be from 1 to 1000. Parameter values are
strict JSON values. Integer values must fit in signed 64-bit range. Documents
are decoded with exact signed 64-bit integer preservation; missing pointers,
overflowing integers, non-string identities, and unsupported values follow the
configured malformed-record policy.

Checkpoints bind the complete ordered mapping and store the continuation used
to open the current page plus the number of documents already handled in that
page. Resume reopens that page and skips only the checkpointed documents.
Cosmos query paging is not a transactional snapshot: source inserts, updates,
deletes, or partition topology changes during a load can change later pages or
the contents replayed after a restart. Use a stable source dataset for
repeatable migration. Diagnostics expose only a hash-derived continuation
identifier, never the full continuation token, access token, authorization
header, account key, or source document.

### Cosmos DB for Apache Gremlin documents

To interpret the backing JSON documents of a Cosmos DB for Apache Gremlin
container automatically, replace explicit `vertices` and `edges` with a
`gremlin` block:

```yaml
source:
  type: cosmos-nosql
  namespace: crm
  cosmos:
    endpoint: https://example.documents.azure.com:443/
    credential: default-azure
    database: graph-source
    pageSize: 100
    gremlin:
      enabled: true
      container: graph
      partitionKeyProperty: partitionKey
      labelPrefix: App
      relationshipTypePrefix: APP_
      maxLabels: 256
      maxProperties: 1024
      maxDiscoveryDocuments: 100000
```

The adapter queries the container through the Cosmos DB for NoSQL endpoint
using `DefaultAzureCredential`; it does not connect to the Gremlin endpoint or
accept account keys. It discovers vertex labels and edge label/endpoint-label
combinations before target admission, sorts them deterministically, then
streams vertices before edges. Prefixes filter source labels without renaming
them. Edges whose endpoint labels are outside the selected vertex labels are
omitted. `maxLabels` bounds both vertex and edge labels, `maxProperties` bounds
the properties interpreted from each document, `maxDiscoveryDocuments` bounds
the documents scanned by each of the vertex and edge discovery queries, and
generated mappings are capped at 1024. Duplicate labels are removed in the
client because the Go SDK's gateway transport cannot execute a cross-partition
`DISTINCT` query.

Vertex `id` and `label` fields, `_value` property wrappers, and flat-schema
properties are interpreted automatically. A single wrapped value becomes one
AGE property value; multiple wrapped values become a list. Edge fields
`_vertexId`/`_vertexLabel` identify the source and `_sink`/`_sinkLabel` plus
`_sinkPartition` identify the target. Edge properties are read from their flat
JSON values. Graph/system fields, all underscore-prefixed fields, and the
configured partition-key property are excluded from AGE properties.
Meta-properties in `_meta` are not migrated.

Cosmos Gremlin IDs are unique only within a logical partition. The adapter
therefore encodes every vertex and edge identity as a JSON pair containing the
partition-key value and element ID. The edge source uses its own partition key;
the edge target uses `_sinkPartition`. Missing or non-primitive partition keys
are malformed records. Trial mode is not supported for automatically
interpreted Gremlin documents: the Go SDK's gateway transport cannot execute
the cross-partition ordering required for deterministic first-N sampling.

Interpretation is rerun for load, resume, and verify. The resolved mappings,
partition-key property, property bound, and generated queries are
fingerprinted, so a discovered label or endpoint mapping change rejects resume
and verification. Discovery and mapping queries are separate cross-partition
reads rather than a transactional snapshot; keep the source stable throughout
the operation.

The interpreted layout is the backing NoSQL document representation, not the
Gremlin wire response. See Microsoft's
[Gremlin JSON format](https://learn.microsoft.com/azure/cosmos-db/gremlin/support#gremlin-wire-format)
and
[partitioning model](https://learn.microsoft.com/azure/cosmos-db/gremlin/partitioning#graph-partitioning-mechanism).

## Trial migrations

Trial mode creates a deterministic, bounded PoC/Evaluation graph without
requiring connector-specific query changes:

```yaml
trial:
  enabled: true
  maxVerticesPerLabel: 1000
  maxVertices: 10000
  maxEdges: 10000
  maxBytes: 64MiB
  includeLabels:
    - Person
    - Organization
```

The limits default to the values shown, except `maxBytes`, which is capped by
`runtime.memoryLimit`. Vertices are selected in configured source order, first
by the per-label limit and then by the total limit. The source is then scanned
for edges whose start and end identities are both among the selected vertices.
This endpoint-closure rule ensures trial mode never creates a dangling edge.
`includeLabels` optionally restricts vertex selection to configured labels.

Trial mode works identically for CSV, PostgreSQL, Neo4j, and Cosmos DB sources.
It is intentionally restricted to `create` and `replace`: sampling an
incremental load could silently produce an incomplete update. Trial jobs are
also non-resumable. After a failure, start a new trial load; removing or changing
the trial block changes the job fingerprint.

Cosmos trial queries must contain `ORDER BY` on a stable unique key, normally
the document ID. The source dataset must remain unchanged for the duration of
the load. This makes first-N selection repeatable across continuation pages.

The load command's JSON result includes selected counts per label, total
vertices, edges and estimated logical record bytes, skipped counts, and every
limit reached. `maxBytes` bounds selected decoded record data rather than
connector page buffers; the vertex identity index is additionally bounded by
`maxVertices`.

## Load modes

The target modes are `create`, `replace`, `append`, and `upsert`. CSV,
PostgreSQL, Neo4j, and Cosmos DB for NoSQL support all four modes. Every edge mapping
in an `upsert` job must provide an external edge identity field or column.
Graph names must follow the supported Apache AGE naming subset: 3–63 UTF-8
bytes, starting with a letter or underscore, ending with a letter, digit, or
underscore, and containing only letters, digits, underscores, dots, and
hyphens.

Incremental jobs make conflict handling explicit. `appendDuplicate` defaults
to `error`; `ignore-identical` permits an append replay only when the existing
identity, endpoints, and properties are identical. Conflicting duplicates are
always rejected. `propertyMode` controls vertex and edge upserts: `replace`
replaces the complete property object, `merge` retains keys omitted by the
source, and `merge-delete-null` also removes keys whose incoming value is null.
Incremental jobs require an active graph generation previously created or
replaced by agefreighter. Admission compares the graph OID, namespace OID, and
every configured label's kind, label ID, relation OID, sequence OID, and
mapping generation with the stored catalog before any data write.

Only one incremental batch may write a graph at a time. A competing job is
rejected without waiting with the stable error code
`AF_INCREMENTAL_CONFLICT`; the failed job can be resumed after the other writer
releases the graph lock. Batch ownership and label ID allocation use separate
job- and label-scoped locks.

When `errors.missingEndpoint` is `defer`, unresolved edges are persisted in a
bounded target-side store rather than Go memory. Upserts are also queued when
an older deferred row has the same external edge identity, preserving source
order even when `missingEndpoint` is `error` or `quarantine`.
`errors.maxDeferredEdges` defaults to 100000 for upsert and whenever deferral
is enabled, and must be positive in those cases. Resolvable rows are drained
transactionally by later incremental batches.
Reaching the limit rolls back the batch deterministically; deferred edges are
never silently discarded.

`replace` requires the public target graph to exist. It loads into a
deterministic job-specific shadow graph, so a failed or interrupted job leaves
the public target unchanged and can resume the admitted shadow. Promotion
verifies the shadow, renames the old target to a job-specific backup, renames
the shadow to the public target, and updates job and generation metadata in one
PostgreSQL transaction. AGE graph and schema OIDs are retained by rename: the
new active generation keeps the former shadow OID and the backup keeps the old
target OID.

The backup is retained after a successful replacement. Remove it explicitly:

```sh
agefreighter cleanup --target job.yaml JOB_ID
```

Cleanup is idempotent after success. It validates the committed replace job,
active-generation OID, backup name, and backup OID before dropping the backup;
it does not require the original full configuration fingerprint to remain
unchanged. The metadata generation is retained for audit after the physical
backup is removed.
