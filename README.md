# agefreighter

agefreighter 2.x is a Go command-line tool for validated, resumable graph
migration from CSV and other delimited files, PostgreSQL, Neo4j, and Azure
Cosmos DB into [Apache AGE](https://age.apache.org/) or PostgreSQL 19's native
SQL/PGQ property graphs. Property graphs represent
entities as vertices, relationships as edges, and their attributes as
properties.

Apache AGE is an open-source PostgreSQL extension that adds graph storage and
openCypher queries while retaining PostgreSQL's relational capabilities.
[Azure Database for PostgreSQL](https://learn.microsoft.com/azure/postgresql/azure-ai/generative-ai-age-overview)
is Microsoft's managed PostgreSQL service and can enable the AGE extension.
The PostgreSQL 19 target stores lossless properties in relational `jsonb`
columns and exposes them through `GRAPH_TABLE`; it does not install AGE and it
does not provide Cypher. This target remains experimental while PostgreSQL 19
is pre-release. Version 2.3.0 is qualified against the digest-pinned official
PostgreSQL 19 Beta 3 image; PostgreSQL 19 GA requires a fresh qualification.

This branch does not preserve the Python API, CLI, configuration, or defaults
from agefreighter 1.x. The 2.x implementation is maintained on `main`; the 1.x
maintenance line remains available on [`release/1.x`](https://github.com/rioriost/agefreighter/tree/release/1.x).
Follow the [1.x to 2.0 migration guide](docs/migration-1.x-to-2.0.md) before
replacing an existing installation.

## Installation

Release archives contain `agefreighter`, `agefreighter-tools`, the project
`LICENSE`, and `THIRD_PARTY_NOTICES.txt`. Verify the downloaded archive against
`checksums.txt` and the GitHub build-provenance attestation before installing
it.

### macOS

Install the checksum-bound Formula from the `rioriost/cask` tap:

```sh
brew install rioriost/cask/agefreighter
```

The Formula selects the notarized `darwin_arm64` archive for Apple silicon or
`darwin_amd64` for Intel. Alternatively, download the matching archive from the
desired [GitHub release](https://github.com/rioriost/agefreighter/releases),
extract it, and install both binaries:

```sh
tar -xzf agefreighter_v2.3.0_darwin_arm64.tar.gz
sudo install -m 0755 agefreighter agefreighter-tools /usr/local/bin/
```

### Linux

Download the `linux_amd64` or `linux_arm64` archive for the host architecture:

```sh
tar -xzf agefreighter_v2.3.0_linux_amd64.tar.gz
sudo install -m 0755 agefreighter agefreighter-tools /usr/local/bin/
```

### Windows

Download `agefreighter_v2.3.0_windows_amd64.zip`, extract
`agefreighter.exe` and `agefreighter-tools.exe`, and place their directory on
`PATH`.

> **Windows signing status:** The Windows binaries in v2.3.0 are intentionally
> provided without an Authenticode signature. The SignPath Foundation
> application was not approved, so Windows code signing remains planned for a
> later release through a future eligible application or another signing
> arrangement. Windows may display an unknown-publisher or SmartScreen warning.
> Verify the archive checksum and GitHub build-provenance attestation before
> use. See the [code signing policy](docs/code-signing-policy.md).

```powershell
Expand-Archive .\agefreighter_v2.3.0_windows_amd64.zip -DestinationPath .\agefreighter
Get-AuthenticodeSignature .\agefreighter\agefreighter.exe
.\agefreighter\agefreighter.exe version
```

For v2.3.0, `Get-AuthenticodeSignature` is expected to report `NotSigned`.

### Build from source

With the Go version declared in `go.mod` installed:

```sh
git clone https://github.com/rioriost/agefreighter.git
cd agefreighter
git checkout v2.3.0
make build VERSION=2.3.0
```

See the [installation guide](docs/reference/installation.md) for archive names,
checksum and provenance verification, and source-build details.

### Visual Studio Code

AGEFreighter 2.3.0 also provides the open-source **AGEFreighter** VS Code
extension. It discovers migration jobs, guides deterministic CLI operations,
renders bounded reports, and optionally lets the user's selected VS Code chat
model explain redacted evidence. Migration execution, checkpoints, and every
target mutation remain in the separately installed Go CLI.

Install [AGEFreighter from the Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=rioriost.agefreighter)
after installing the CLI, then open the AGEFreighter activity-bar view. See the
[VS Code extension guide](docs/reference/vscode-extension.md) for workspace
trust, remote environments, AI privacy boundaries, and local VSIX installation.

## Quick Usage

- `agefreighter`: production migration CLI
- `agefreighter-tools`: fixtures, diagnostics, AI-assisted conversion, and benchmarks

Start with the validated example for the source being migrated, copy it to
`job.yaml`, and replace its source mappings, credential references, and target:

| Source | Start from | Usage notes |
|---|---|---|
| CSV or other delimited files | [`csv.yaml`](internal/config/testdata/valid/csv.yaml) | Map stable vertex IDs and edge endpoints by column. Comma, tab, or any other supported single-code-point delimiter can be selected globally or per file. Optional [explicit property types](docs/reference/csv-property-types.md) preserve numeric, boolean and array values. |
| PostgreSQL | [`postgresql.yaml`](internal/config/testdata/valid/postgresql.yaml) | Reference the source DSN through an environment variable and provide ordered SQL vertex and edge queries. `copy` is the default streaming mode; `keyset` provides durable resume for suitable append-only sources. |
| Neo4j | [`neo4j.yaml`](internal/config/testdata/valid/neo4j.yaml) | Reference Bolt credentials and provide ordered Cypher mappings with stable keys and external IDs, or use [`neo4j-discovery.yaml`](internal/config/testdata/valid/neo4j-discovery.yaml) for bounded automatic graph discovery. |
| Cosmos DB for NoSQL | [`cosmos.json`](internal/config/testdata/valid/cosmos.json) | Authenticate with `DefaultAzureCredential`, use parameterized cross-partition queries, and map fields with JSON Pointers. For Cosmos Gremlin backing documents, use [`cosmos-gremlin.json`](internal/config/testdata/valid/cosmos-gremlin.json) to discover and interpret the graph automatically. |

Credentials are references to environment variables or files; literal secrets
are rejected. Export the variables named by the selected job, then validate and
inspect the execution plan without connecting to the source or target:

```sh
agefreighter validate job.yaml
agefreighter plan job.yaml
```

CSV, PostgreSQL, Neo4j, and Cosmos DB for NoSQL sources support `create`, atomic
`replace`, `append`, and `upsert`. Run the migration and verify the committed
graph:

```sh
agefreighter load job.yaml
agefreighter status --target job.yaml JOB_ID
agefreighter verify --target job.yaml JOB_ID
agefreighter report --target job.yaml JOB_ID
```

For PostgreSQL 19 SQL/PGQ, select `target.type:
postgresql-property-graph`, set `target.schema`, and use a PostgreSQL 19 target
qualified in the [compatibility matrix](docs/reference/compatibility.md).
All four load modes use the same checkpoint and digest contracts as above;
verification additionally checks relational constraints and directed and
undirected `GRAPH_TABLE` traversal. Start from the validated
[`postgresql-property-graph.yaml`](internal/config/testdata/valid/postgresql-property-graph.yaml)
example.

Diagnose target readiness without migrating metadata or changing graph data:

```sh
agefreighter doctor --target job.yaml
agefreighter doctor --target job.yaml --format markdown --output doctor.md
```

Profile the configured source without opening or changing the Apache AGE target:

```sh
agefreighter profile job.yaml
agefreighter profile --mode exact --format markdown job.yaml
```

The default profile is a bounded prefix sample. It reports aggregate mapping,
type, null, cardinality, width, endpoint, capacity, and connector-telemetry
signals without including source values, record identities, query text,
credentials, or continuation tokens. See the
[operations guide](docs/reference/operations.md#bounded-source-profiles).

Inspect bounded target evidence and review deterministic post-load
recommendations without changing the target:

```sh
agefreighter optimize --target job.yaml
```

Only the explicit `--apply-analyze` flag permits the optimizer to run `ANALYZE`, and
then only for revalidated active label relations and an exact metadata
allowlist under statement, lock, and operation deadlines. The optimizer never
applies index DDL. See
[recommendation-first target optimization](docs/reference/operations.md#recommendation-first-target-optimization).

Statically check local Cypher without connecting to a database or service:

```sh
agefreighter-tools check-cypher queries.cypher --format json
agefreighter-tools check-cypher queries.cypher --strict
```

The same bounded analyzer can add deduplicated workload evidence to an
optimizer report with repeated `--queries FILE` flags. Query text and parameter
values are never included in either report.

If a load fails after committing a resumable checkpoint, continue the same job:

```sh
agefreighter resume --job job.yaml JOB_ID
```

For a committed `replace`, remove the retained backup graph when it is no
longer needed:

```sh
agefreighter cleanup --target job.yaml JOB_ID
```

For bounded PoC and evaluation loads, add a `trial` block to a `create` or
`replace` job. Trial mode selects bounded vertices and emits only edges whose
endpoints were selected; trial jobs are intentionally non-resumable. See the
[configuration reference](docs/reference/configuration.md#trial-migrations).

Neo4j jobs can either define explicit Cypher mappings or use bounded,
read-only graph discovery with application-owned stable key and identity
properties. See the
[Neo4j configuration reference](docs/reference/configuration.md#neo4j-options).

Cosmos DB for Apache Gremlin backing documents can be interpreted directly
through the NoSQL endpoint, including property wrappers and partition-aware
vertex/edge identities. See the
[Cosmos configuration reference](docs/reference/configuration.md#cosmos-db-for-apache-gremlin-documents).

Inspect a validated job's source mappings and target configuration without
connecting to either service:

```sh
agefreighter-tools inspect job.yaml
```

Inspection output is deterministic JSON. It identifies configured source
locations, identity/resume fields, endpoint mappings, and target behavior while
omitting query text, connection strings, credentials, and credential reference
names.

Convert a Gremlin traversal to openCypher without executing either query:

```sh
OPENAI_API_KEY='...' \
  agefreighter-tools convert-gremlin --input query.gremlin
```

Normalize one or more `benchmark-age-copy` result streams into a canonical JSON
or Markdown report:

```sh
agefreighter-tools benchmark-age-copy > run-1.json
agefreighter-tools benchmark-report run-1.json run-2.json
agefreighter-tools benchmark-report --format markdown < results.jsonl
```

See the [tools reference](docs/reference/tools.md) for conversion and inspection
contracts, benchmark report schemas, validation, aggregation, and input limits.

Operational logging and OTLP trace export are disabled by default. See the
[observability reference](docs/reference/observability.md) and
[compatibility matrix](docs/reference/compatibility.md) for the supported
runtime contract.

Production procedures are in the [operations guide](docs/reference/operations.md).

See the [configuration reference](docs/reference/configuration.md), the
[architecture research](docs/design/agefreighter-2.0-research.md) and the
[implementation plan](docs/design/agefreighter-2.0-implementation-plan.md) for
the delivery sequence.

## Requirements

- Go 1.27 or newer
- Apple Container 1.0 for local database integration tests
- Docker on Linux for CI database integration tests
- Azure CLI and Azure Developer CLI for Cosmos DB integration tests

## Performance

The review-gated [production simulation](production-simulation/README.md)
qualifies Neo4j 4.4.48 and 5.26.30 migrations at up to 160 million vertices
and 400 million edges against PostgreSQL 18 with Apache AGE 1.7 on Azure.
Large runs are manual and are never part of ordinary pull-request CI.

The release benchmark compares end-to-end CSV `create` throughput using the
same 20,200-row countries corpus, Apple M4 Max host, and pinned PostgreSQL 17 /
Apache AGE 1.6 environment. Throughput is the number of rows verified in AGE
divided by wall-clock load time.

| Version | Implementation | Reported throughput | Relative throughput |
|---|---|---:|---:|
| 1.0.36 | Python | 54,595 rows/s | 1.00x |
| 2.0.0 | Go | 118,832 rows/s | **2.18x** |

The 1.0.36 result is the median of five measured loads after one warm-up; the
2.0.0 result averages three complete loads. This is a whole-load comparison,
not a language microbenchmark: the 2.0.0 path also persists target identities
and durable checkpoints. See the [1.0.36 baseline](docs/benchmarks/1x-baseline.md)
and [2.0.0 release performance gates](docs/benchmarks/2x-release-gates.md) for
the full procedure, environment, and release thresholds.

## Development

Install the pinned development tools:

```sh
make install-tools
```

Run the local quality gates:

```sh
make check
```

Run the full test and coverage gate:

```sh
make check-full
```

With the pinned services running and their connection variables exported, run
the explicit release integration hooks:

```sh
make test-connectors-local
make test-release-integration
make test-diagnostics-race
make bench-release
```

Build both binaries:

```sh
make build
```

Start the pinned local AGE, PostgreSQL, and Neo4j fixtures:

```sh
make dev-up
```

See the [development database guide](scripts/dev/README.md) for the lifecycle,
ports, image digests, and reset safety boundary.

The unexcluded repository-wide statement coverage threshold is 80%. Coverage is a release gate,
not a substitute for race, fuzz, contract, container integration, and recovery
tests.

## Versioning

Development builds report `dev`. Release builds inject the version, commit, and
build date through Go linker variables.

```sh
go run ./cmd/agefreighter version
go run ./cmd/agefreighter-tools version
```

## License

[MIT](LICENSE)
