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
snapshot-exporting transaction remains open until the iterator closes. Reader
transactions import that snapshot before executing any source query, so
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
session-pooling mode. The coordinator supports bounded concurrent snapshot
readers, while the 2.0.0 iterator processes configured mappings sequentially
to preserve vertex-before-edge order.

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

## Load modes

The target modes are `create`, `replace`, `append`, and `upsert`. CSV,
PostgreSQL, and Cosmos DB for NoSQL support all four modes. Every edge mapping
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
