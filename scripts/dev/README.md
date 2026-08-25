# Development database environment

`dev.sh` provides the same lifecycle contract through Apple Container on macOS
and Docker on Linux CI. It manages only the fixed `agefreighter-*` resources
listed below. It never invokes a prune command.

## Service matrix

| Service | Image index digest | Local ports | Volume |
| --- | --- | --- | --- |
| PostgreSQL 17 + AGE 1.6.0 | `apache/age@sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be` | `127.0.0.1:55432` | `agefreighter-pg17-age160-data` |
| PostgreSQL 17.6 source | `postgres@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94` | `127.0.0.1:55433` | `agefreighter-postgres17-source-data` |
| Neo4j 5.26.29 Community | `neo4j@sha256:89d577f2e49606de76441eca8cf7a0fe88e594cbaac4d2a3d86c6e59676e2b1e` | `127.0.0.1:57687`, `127.0.0.1:57474` | `agefreighter-neo4j526-data` |

Verified platform manifests:

| Image | `linux/arm64` | `linux/amd64` |
| --- | --- | --- |
| AGE | `sha256:7ae38a9f8a908cff60a63a31b57b12094f5d47036409abde38bca7ac52cae014` | `sha256:2286e5eb88ae0ea8e079d60c9fd9bded2ba9b7c2ff347d7b3739c58d43cdba6e` |
| PostgreSQL | `sha256:9b9fb55f7e3b2149854def33c728b781dc44d1c5e86492ad62912a527ae234b3` | `sha256:747d5ed1fdeeb124b880fbe3d7c6557d2c4064ae41d6b6297d417882effce4be` |
| Neo4j | `sha256:a58f3320cb0112b71df549bbebe42b75236b0dd365df551d1bf16c127f8414f7` | `sha256:f0d31a1add53b219f8b1e5897301128620b142e15e68e67c71b3ef1b483c4c67` |

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

`make coverage` uses this local DSN by default and therefore requires
`make dev-up`. This keeps the 90% gate based on the live AGE contract rather
than counting database adapter code as excluded.

Generate deterministic CSV source data without overwriting an existing output
directory:

```sh
go run ./cmd/agefreighter-tools generate fixture --output .local/fixture
go run ./cmd/agefreighter-tools generate benchmark \
  --output .local/benchmark --vertices 100000 --edges 500000 --seed 1
```
