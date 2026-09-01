# Acceptance and stop criteria

## Required artifacts

Every run produces a versioned summary containing:

- git commit, agefreighter version and checksum;
- Azure subscription alias, region, zone, resource IDs, SKUs, disk settings,
  HA mode, and resource tags, without credentials;
- Neo4j version, image digest, JVM, heap/page-cache settings, store/index size,
  and `memory-recommendation` output;
- PostgreSQL and AGE versions, relevant server parameters, and extension list;
- fixture manifest root, phase, seed, counts, and shard count;
- normalized load plan, job ID, report, doctor, optimize, and verify output;
- fault timeline and checkpoint IDs;
- monitoring export, elapsed time, peak values, forecast, and Azure cost.

Raw logs and metrics may contain hostnames or connection metadata and remain in
the ignored raw-results location. A redacted summary is reviewed before commit.

## P0 correctness gate

P0 validates the Azure harness and exact-count path. It passes only when:

- exact total and per-label/type counts match;
- rejected records, failed batches, duplicates, and missing endpoints are zero;
- every executed built-in count and bounded-integrity field has no failure;
- any `verify` incomplete result is caused only by the documented 1,000-row
  integrity bound, and the affected labels are recorded in the summary; and
- source counts are unchanged after both migrations.

The bounded-integrity exception does not authorize P2. The independent
canonical range-digest verifier must be implemented and proven during P1, as
specified in `dataset.md` and `test-plan.md`.

## P1-P4 correctness gates

All are hard requirements:

- exact total and per-label/type counts;
- zero rejected records, failed batches, duplicates, and missing endpoints;
- built-in count verification passes and bounded integrity reports no failed
  field; any bound-related incomplete coverage is closed by the independent
  canonical range-digest verifier;
- every canonical key-range digest and final Merkle root match;
- the recovery run produces the same root as the clean run;
- checkpoints advance monotonically and resume does not skip or duplicate;
- no source write occurs between discovery and final verification.

## Provisional performance gates

P2 freezes the final P3 service-level objective. Until then:

- a P3 clean run should finish in 24 hours or less;
- P3 agefreighter RSS remains at or below 4 GiB with no upward size-related
  trend; earlier phases and non-P3 release gates retain their 2 GiB limit;
- no process swaps or is terminated for memory pressure;
- P3 sustained throughput is at least 70% of P2 after warmup;
- the final throughput quintile is at least 80% of the first quintile;
- p99 committed-batch latency is no more than three times its median;
- recovery begins within 15 minutes, excluding replay after the last checkpoint;
- target storage remains below 80% utilization.

Measurements are sampled every 15-30 seconds. Loader metrics include records
and bytes per second, batch latency, RSS, CPU, GC, failures, and checkpoint age.
Neo4j metrics include CPU, heap, GC, page cache, disk latency, Bolt latency, and
query plans. PostgreSQL metrics include CPU, IOPS, throughput, latency, storage,
WAL, checkpoints, temporary files, locks, relation/index sizes, and metadata
table sizes. Network throughput, retransmits, and RTT are captured separately.

## Automatic stop conditions

Stop the current run without deleting evidence when any condition is met:

- target or source storage reaches 80%;
- no committed checkpoint for 15 minutes while work remains;
- a reject, missing endpoint, duplicate, checksum error, or unexpected source
  mutation is detected;
- any host swaps, hits OOM, or loses its durable volume;
- sustained I/O throttling reduces throughput below half the P2 baseline for
  five minutes;
- projected P3 duration or cost exceeds twice the approved P2 forecast;
- the Azure resource identity, region, zone, version, or SKU differs from the
  reviewed deployment manifest.

Stopping is not cleanup. Preserve logs, manifests, checkpoint state, and
metrics until the failure has been reviewed.
