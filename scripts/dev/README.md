# Development database environment

`dev.sh` provides the same lifecycle contract through Apple Container on macOS
and Docker on Linux CI. It manages only the fixed `agefreighter-*` resources
listed below. It never invokes a prune command.

## Service matrix

| Service | Image index digest | Local ports | Volume |
| --- | --- | --- | --- |
| PostgreSQL 17 + AGE 1.6.0 | `apache/age@sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be` | `127.0.0.1:55432` | `agefreighter-pg17-age160-data` |
| PostgreSQL 17.6 source | `postgres@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94` | `127.0.0.1:55433` | `agefreighter-postgres17-source-data` |
| Neo4j 5.26.30 Community | `neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d` | `127.0.0.1:57687`, `127.0.0.1:57474` | `agefreighter-neo4j526-data` |

Verified platform manifests:

| Image | `linux/arm64` | `linux/amd64` |
| --- | --- | --- |
| AGE | `sha256:7ae38a9f8a908cff60a63a31b57b12094f5d47036409abde38bca7ac52cae014` | `sha256:2286e5eb88ae0ea8e079d60c9fd9bded2ba9b7c2ff347d7b3739c58d43cdba6e` |
| PostgreSQL | `sha256:9b9fb55f7e3b2149854def33c728b781dc44d1c5e86492ad62912a527ae234b3` | `sha256:747d5ed1fdeeb124b880fbe3d7c6557d2c4064ae41d6b6297d417882effce4be` |
| Neo4j | `sha256:77ad51ca3579a345bd5ea842234480b87319da4c9281b410d3d305b856b4e905` | `sha256:b7deccb0181b9ad9cbeaf1728d099066fe30210e42bf1e1295da1c587eee4012` |

The fixed development-only password is `agefreighter_dev_only`. Services are
published only on localhost and must not be exposed to another network.

## Commands

```sh
make dev-pull
make dev-up
make dev-status
make dev-smoke
make dev-down
make dev-reset
```

`dev-up` is idempotent and initializes deterministic fixtures after readiness
checks. `dev-down` preserves all volumes. `dev-reset` deletes only the three
documented containers and volumes, then recreates the deterministic fixtures.

The runtime and platform can be selected explicitly:

```sh
DEV_RUNTIME=apple DEV_PLATFORM=linux/arm64 ./scripts/dev/dev.sh up
DEV_RUNTIME=docker DEV_PLATFORM=linux/amd64 ./scripts/dev/dev.sh up
```

## AGE adapter integration tests

With the development services running, execute the live AGE adapter contract
tests with:

```sh
AGEFREIGHTER_AGE_TEST_DSN='postgres://agefreighter:agefreighter_dev_only@127.0.0.1:55432/agefreighter?sslmode=disable' \
  go test -v ./internal/age
```

Integration DSNs and connector credentials have no Makefile defaults: an
unset environment causes the corresponding Go integration test to print an
explicit skip reason. After `make dev-up`, source `scripts/dev/services.sh`,
export the AGE, PostgreSQL, and Neo4j variables as shown in CI, and run
`make coverage`. This keeps the 80% unexcluded repository-wide gate based on
the live connector contracts rather than counting database code as excluded
or pretending an absent service passed.

Generate deterministic CSV source data without overwriting an existing output
directory:

```sh
go run ./cmd/agefreighter-tools generate fixture --output .local/fixture
go run ./cmd/agefreighter-tools generate benchmark \
  --output .local/benchmark --vertices 100000 --edges 500000 --seed 1
```

The Apple Container bulk-write harness records elapsed time, PostgreSQL
container CPU, WAL bytes, peak target RSS, and peak client RSS for one workload
and strategy:

```sh
mkdir -p .local/benchmarks
AGEFREIGHTER_AGE_TEST_DSN='postgres://agefreighter:agefreighter_dev_only@127.0.0.1:55432/agefreighter?sslmode=disable' \
  ./scripts/bench/age-copy.sh \
  edges direct-text 500000 5 .local/benchmarks/edges-direct.jsonl
```

The output path must not exist. Supported strategies are `direct-text`,
`staged-binary`, and `plain-relational`. The harness restarts only the fixed AGE
development container before each trial so kernel `memory.peak` starts from a
comparable cold baseline; its named volume is preserved. Reported target
memory is total cgroup memory, while client memory is peak process RSS.

## PostgreSQL 19 property graph development target

The experimental 2.2.0 target has a separate Apple Container harness. It pins
the official PostgreSQL 19 Beta 3 image index digest, verifies the resolved
digest before starting, and manages only `agefreighter-pg19-sqlpgq`:

```sh
./scripts/dev/pggraph-apple-container.sh up
./scripts/dev/pggraph-apple-container.sh status
make test-pggraph-apple
./scripts/dev/pggraph-apple-container.sh down
```

`down` preserves the dedicated container and database. The harness does not
delete images, volumes, or unrelated Apple Container resources. A newer beta
or release candidate intentionally fails the digest check until its SQL/PGQ
behavior is reviewed and the recorded digest is updated.

Run the complete-path property-graph benchmark against the same pinned target:

```sh
AGEFREIGHTER_PGGRAPH_TEST_DSN='postgres://...' \
  make bench-pggraph PGGRAPH_BENCH_PROFILE=small \
  PGGRAPH_BENCH_TRIALS=3 \
  PGGRAPH_BENCH_OUTPUT=.local/pggraph-small.txt
```

`small` loads 10,000 vertices and 25,000 edges; `medium` loads 100,000 and
250,000. The explicit `production` profile is 160,000,000 and 400,000,000 and
also requires `PGGRAPH_BENCHMARK_PRODUCTION_ACK=160000000-400000000`. This
guard prevents an accidental multi-hundred-million-row local run. Benchmark
outputs are local evidence and are never overwritten.
