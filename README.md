# agefreighter

agefreighter 2.x is a greenfield Go implementation for validated, resumable
graph migration into [Apache AGE](https://age.apache.org/).

This branch does not preserve the Python API, CLI, configuration, or defaults
from agefreighter 1.x. The 1.x implementation remains on the `main` branch.

## Commands

- `agefreighter`: production migration CLI
- `agefreighter-tools`: fixtures, diagnostics, and benchmarks

The versioned load-job configuration, offline validation, and static planning
commands are available:

```sh
go run ./cmd/agefreighter validate ./path/to/job.yaml
go run ./cmd/agefreighter plan ./path/to/job.yaml
```

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
