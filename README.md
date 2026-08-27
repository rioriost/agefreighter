# agefreighter

agefreighter 2.x is a greenfield Go implementation for validated, resumable
graph migration into [Apache AGE](https://age.apache.org/).

This branch does not preserve the Python API, CLI, configuration, or defaults
from agefreighter 1.x. The 1.x implementation remains on the `main` branch.

## Commands

- `agefreighter`: production migration CLI
- `agefreighter-tools`: fixtures, diagnostics, AI-assisted conversion, and benchmarks

The versioned load-job configuration, offline validation, and static planning
commands are available:

```sh
go run ./cmd/agefreighter validate ./path/to/job.yaml
go run ./cmd/agefreighter plan ./path/to/job.yaml
```

CSV, PostgreSQL, Neo4j, and Cosmos DB for NoSQL sources support `create`, atomic
`replace`, `append`, and `upsert` through the durable lifecycle commands:

```sh
agefreighter load job.yaml
agefreighter resume --job job.yaml JOB_ID
agefreighter status --target job.yaml JOB_ID
agefreighter verify --target job.yaml JOB_ID
agefreighter cleanup --target job.yaml JOB_ID
```

`cleanup` applies only to committed `replace` jobs and removes the retained
backup graph without removing the active graph.

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

Release archives, checksum and provenance verification, Homebrew installation,
and source builds are covered by the
[installation guide](docs/reference/installation.md). Production procedures
are in the [operations guide](docs/reference/operations.md).

See the [configuration reference](docs/reference/configuration.md), the
[architecture research](docs/design/agefreighter-2.0-research.md) and the
[implementation plan](docs/design/agefreighter-2.0-implementation-plan.md) for
the delivery sequence.

## Requirements

- Go 1.27 or newer
- Apple Container 1.0 for local database integration tests
- Docker on Linux for CI database integration tests
- Azure CLI and Azure Developer CLI for Cosmos DB integration tests

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
