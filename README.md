# agefreighter

agefreighter 2.x is a Go command-line tool for validated, resumable graph
migration from CSV and other delimited files, PostgreSQL, Neo4j, and Azure
Cosmos DB into [Apache AGE](https://age.apache.org/). Property graphs represent
entities as vertices, relationships as edges, and their attributes as
properties.

Apache AGE is an open-source PostgreSQL extension that adds graph storage and
openCypher queries while retaining PostgreSQL's relational capabilities.
[Azure Database for PostgreSQL](https://learn.microsoft.com/azure/postgresql/azure-ai/generative-ai-age-overview)
is Microsoft's managed PostgreSQL service and can enable the AGE extension.

This branch does not preserve the Python API, CLI, configuration, or defaults
from agefreighter 1.x. The 1.x implementation remains on the `main` branch.

## Installation

Release archives contain both `agefreighter` and `agefreighter-tools`. Verify
the downloaded archive against `checksums.txt` and the GitHub build-provenance
attestation before installing it.

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
tar -xzf agefreighter_v2.0.0_darwin_arm64.tar.gz
sudo install -m 0755 agefreighter agefreighter-tools /usr/local/bin/
```

### Linux

Download the `linux_amd64` or `linux_arm64` archive for the host architecture:

```sh
tar -xzf agefreighter_v2.0.0_linux_amd64.tar.gz
sudo install -m 0755 agefreighter agefreighter-tools /usr/local/bin/
```

### Windows

Download `agefreighter_v2.0.0_windows_amd64.zip`, extract
`agefreighter.exe` and `agefreighter-tools.exe`, and place their directory on
`PATH`. Release executables carry a timestamped Windows Authenticode signature
provided through SignPath Foundation. See the
[code signing policy](docs/code-signing-policy.md):

Prereleases published while SignPath Foundation enrollment is pending may omit
the Windows archive. Official Windows binaries are never published unsigned;
check the assets and notes for the release you are installing.

```powershell
Expand-Archive .\agefreighter_v2.0.0_windows_amd64.zip -DestinationPath .\agefreighter
Get-AuthenticodeSignature .\agefreighter\agefreighter.exe
.\agefreighter\agefreighter.exe version
```

### Build from source

With the Go version declared in `go.mod` installed:

```sh
git clone https://github.com/rioriost/agefreighter.git
cd agefreighter
git checkout v2.0.0
make build VERSION=2.0.0
```

See the [installation guide](docs/reference/installation.md) for archive names,
checksum and provenance verification, and source-build details.

## Quick Usage

- `agefreighter`: production migration CLI
- `agefreighter-tools`: fixtures, diagnostics, AI-assisted conversion, and benchmarks

Start with the validated example for the source being migrated, copy it to
`job.yaml`, and replace its source mappings, credential references, and Apache
AGE target:

| Source | Start from | Usage notes |
|---|---|---|
| CSV or other delimited files | [`csv.yaml`](internal/config/testdata/valid/csv.yaml) | Map stable vertex IDs and edge endpoints by column. Comma, tab, or any other supported single-code-point delimiter can be selected globally or per file. |
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
```

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

The merged statement coverage threshold is 90%. Coverage is a release gate,
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
