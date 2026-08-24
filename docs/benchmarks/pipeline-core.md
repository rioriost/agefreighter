# Pipeline core benchmark

This benchmark measures bounded-pipeline scheduling, batching, memory
accounting, and an in-memory fake sink. It excludes decoding, transformation,
network, database, and durability costs.

## Milestone 2 baseline

- Go: 1.27.0
- OS/architecture: macOS, arm64
- CPU: Apple M4 Max
- Records per iteration: 1,000
- Record payload accounting: 128 bytes
- Batch: 1,000 records / 128,000 payload bytes

Command:

```sh
go test -run '^$' -bench BenchmarkRunner -benchtime=1000x -count=3 \
  -benchmem ./internal/pipeline
```

Observed range:

| Metric | Result |
| --- | ---: |
| Time per 1,000 records | 143,288–144,764 ns |
| Approximate records/second | 6.9 million |
| Allocated bytes per iteration | 157,997–158,051 B |
| Allocations per iteration | 52 |

The result is a development baseline for detecting pipeline-core regressions,
not an end-to-end throughput claim. Connector and Apache AGE benchmarks begin
after the local database environment is pinned.
