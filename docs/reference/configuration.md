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

The target modes are `create`, `replace`, `append`, and `upsert`. The current
runtime implements CSV and Cosmos DB for NoSQL `create` and `replace`; `append`
and `upsert` remain reserved for a later milestone. Every edge mapping in an
`upsert` job must provide an external edge identity field or column. Graph
names must follow the supported Apache AGE naming subset: 3–63 UTF-8 bytes,
starting with a letter or underscore, ending with a letter, digit, or
underscore, and containing only letters, digits, underscores, dots, and
hyphens.

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
