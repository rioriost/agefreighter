# agefreighter 2.x CSV `create` benchmark

This report records the Milestone 5 CSV-to-AGE performance and bounded-memory
gates. It uses the same host, pinned AGE image, PostgreSQL durability settings,
and preserved countries corpus documented in
[`1x-baseline.md`](1x-baseline.md).

## Correctness-equivalent path

The measured `Load` lifecycle includes configuration loading, connection and
capability admission, graph and label creation, streaming CSV parsing, AGE
COPY, target-side vertex and edge identities, endpoint resolution, durable
batch checkpoints, and graph-generation activation. Benchmark cleanup is
outside the timed region.

The optimized path retains:

- canonical JSON property encoding with typed-property fallback
- arbitrary single-rune delimiters and custom quote/escape parsing
- an RFC 4180 fast path for the common double-quote form
- staged AGE binary COPY
- direct typed PostgreSQL binary COPY for identity and endpoint staging
- set-based endpoint resolution without an unbounded Go identity map
- transactionally atomic AGE rows, identities, and checkpoints
- label-generation validation and `FOR KEY SHARE` locking before identity COPY

Identity provenance is normalized through `label_generation`. Identity INSERTs
are an internal sink operation; their admitted generation is locked in the data
transaction. UPDATE validation and label-generation cleanup remain enforced by
metadata triggers.

## Countries corpus result

Command:

```sh
make bench-csv BENCHTIME=9x
```

Each reported run is the mean of nine complete loads. Three independent command
invocations produced:

| Invocation | Time/load | Throughput | Allocated/load | Allocations/load |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 178.526 ms | 113,149 rows/s | 28.22 MB | 224,803 |
| 2 | 181.252 ms | 111,447 rows/s | 28.19 MB | 224,791 |
| 3 | 188.081 ms | 107,400 rows/s | 28.18 MB | 224,784 |
| **Median** | **181.252 ms** | **111,447 rows/s** | **28.19 MB** | **224,791** |

The committed 1.x median is 54,595 rows/s. The 2.x median is **2.04x** the 1.x
baseline and exceeds the required 109,190 rows/s gate by 2.1%.

## Input-size and RSS scaling

`bench-csv-scale` generates its CSV before the timed region and loads it with a
64 MiB pipeline memory limit and 10,000-row batches.

```sh
make bench-csv-scale SCALE_ROWS=20000
make bench-csv-scale SCALE_ROWS=200000
```

| Rows | Time/load | Throughput | Total allocated/load |
| ---: | ---: | ---: | ---: |
| 20,000 | 196.079 ms | 102,000 rows/s | 20.86 MB |
| 200,000 | 937.668 ms | 213,295 rows/s | 205.75 MB |

Peak process RSS was measured in separate processes:

```sh
/usr/bin/time -l sh -c 'make bench-csv-scale SCALE_ROWS=20000 >/dev/null'
/usr/bin/time -l sh -c 'make bench-csv-scale SCALE_ROWS=200000 >/dev/null'
```

| Rows | Maximum RSS |
| ---: | ---: |
| 20,000 | 200.0 MiB |
| 200,000 | 194.7 MiB |

The measurement includes the Go test/tooling process, so it is not a
loader-only absolute footprint. It is suitable for the scaling gate: a 10x
input increase did not increase peak client RSS. Total allocations grow with
the work performed, while live memory remains bounded by batch and channel
limits.
