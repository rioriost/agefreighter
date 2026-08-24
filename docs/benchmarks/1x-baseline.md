# agefreighter 1.x CSV-to-AGE baseline

This is the reproducible end-to-end baseline for the agefreighter 2.x CSV load
performance gate. It includes 1.x CSV parsing, mapping, temporary CSV
generation, graph replacement, label creation, PostgreSQL COPY, sequence
updates, and cleanup.

## Source and environment

| Item | Value |
| --- | --- |
| agefreighter source | `20cf854510d6bfa4295d0901a313219ce855d145` (`main`, version 1.0.36) |
| Checkout | Detached, independent Git worktree |
| Python | CPython 3.13.14 |
| Dependency manager | uv 0.12.5 |
| Host | Apple M4 Max, 128 GiB RAM |
| Host OS | macOS 26.6.2 (25G83), arm64 |
| Container runtime | Apple Container 1.0.0 |
| Container platform | `linux/arm64` |
| PostgreSQL | 17.7 |
| Apache AGE | 1.6.0 |
| Measurement date | 2026-08-24 UTC |

The AGE image was
`apache/age@sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be`;
its selected arm64 manifest was
`sha256:7ae38a9f8a908cff60a63a31b57b12094f5d47036409abde38bca7ac52cae014`.

### Locked-environment repair

`uv sync --frozen --extra test --group dev` reproduces the committed lock, but
the 1.0.36 project metadata incorrectly includes `psycopg-binary==3.3.3`
without the importable `psycopg` package. The unmodified locked environment
fails before connecting with `No module named 'psycopg'`.

The benchmark therefore applies exactly one runtime repair:

```sh
uv pip install 'psycopg==3.3.3'
```

This matches the locked binary package version. No 1.x source or configuration
file was modified.

## Dataset manifest

Configuration:
`docs/configs/csv/single_edge_multiple_nodes.json`

| File | Data rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| configuration | — | — | `64dea84b576fc005ab4b7a94ea770e76e5b6345ab89e3f0cfdb2201f78fb410f` |
| `data/countries/country.csv` | 200 | 18,277 | `ddfc7e373631497e11cbb9a94d2ec38df4ca21c5ee2f8f1b0d4ff184af39a113` |
| `data/countries/city.csv` | 10,000 | 476,736 | `e4f53c6bc5dc4a1a2aa0f6cbb085d27b05475ba8d24bd3166d9c420d3df7628b` |
| `data/countries/has_country_city.csv` | 10,000 | 652,499 | `8d66bb3bebf053eb7ba6ef1b95d0a640674d03187bdfbad29da241a6269b9568` |
| **Total** | **20,200** | **1,147,512** | — |

The CSV files are preserved under `testdata/legacy-baseline/countries/` on the
2.x branch. File hashes match the 1.x source worktree.

## Procedure

The pinned development AGE container was reset and initialized by
`scripts/dev/dev.sh`. One unmeasured warm-up load preceded five measured loads.
Every invocation used the same graph name; 1.x dropped and recreated that graph
inside the measured interval.

```sh
env -u VIRTUAL_ENV /usr/bin/time -l \
  uv run agefreighter \
  --graphname benchmark_1x \
  --pg-con-str \
    'host=127.0.0.1 port=55432 dbname=agefreighter user=agefreighter password=agefreighter_dev_only' \
  --pg-min-connections 1 \
  --pg-max-connections 4 \
  load \
  --source-type csv \
  --config docs/configs/csv/single_edge_multiple_nodes.json \
  --chunk-size 10000
```

After the final load, direct table counts were:

| AGE label | Rows |
| --- | ---: |
| `Country` | 200 |
| `City` | 10,000 |
| `has` | 10,000 |
| **Total** | **20,200** |

## Results

Throughput is verified AGE rows divided by wall-clock time. Peak RSS is the
maximum resident set size reported by macOS `/usr/bin/time -l` for the 1.x
Python process; it excludes the AGE container VM.

| Run | Wall time | Throughput | Peak RSS |
| ---: | ---: | ---: | ---: |
| 1 | 0.36 s | 56,111 rows/s | 65.42 MiB |
| 2 | 0.37 s | 54,595 rows/s | 65.47 MiB |
| 3 | 0.44 s | 45,909 rows/s | 65.28 MiB |
| 4 | 0.49 s | 41,224 rows/s | 65.45 MiB |
| 5 | 0.36 s | 56,111 rows/s | 65.27 MiB |
| **Median** | **0.37 s** | **54,595 rows/s** | **65.42 MiB** |

The release gate compares a correctness-equivalent 2.x load against the median:
at least **109,190 rows/s** is required for the 2x target, with **272,975
rows/s** as the 5x stretch target. Larger generated datasets will supplement
this small-corpus baseline before final release benchmarking.
