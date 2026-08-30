# P3 reviewed run sheet

## Authorization and deadline

The user authorized P3 execution and its forecast budget on 2026-08-31. P3
must complete during the week ending 2026-09-06. The reviewed live-operation
window is 72 hours and the cost ceiling is 225 USD. This authorization covers
deployment, source preparation, four migration runs, the controlled faults
below, full verification, monitoring, evidence retention, and stopping the
resources. It does not authorize P4 or deletion of resources or evidence.

P3 uses a new `rg-afps-p3-20260831` resource group in Japan East zone 1. All
VMs and PostgreSQL remain private and in the same region and zone.

| Role | Frozen P3 value | P2 basis |
| --- | --- | --- |
| Neo4j source, each | `Standard_E8bds_v5`; 24 GiB heap / 28 GiB page cache | Frozen P2 source configuration |
| Source data disk, each | 1,024 GiB Premium SSD v2; 10,000 IOPS / 750 MB/s | 303 GB P3 projection plus import headroom |
| Loader | `Standard_D8ds_v5`; 256 GiB OS disk | P2 CPU 44.52% max, RSS below 183 MiB |
| Flexible Server | PostgreSQL 18 / AGE 1.7; `Standard_E8ds_v5`; SameZone HA | P2 CPU 30.27%, memory 33.34% max |
| Target storage | 4,096 GiB, autogrow disabled | Four retained ~425 GB P3 databases stay below 80% |

The migration configuration is frozen at `fetchRows: 5000`,
`batchRows: 20000`, `batchBytes: 64MiB`, a 2 GiB loader memory limit, and
create mode. P3 does not perform tuning or replacement.

## Required runs

The logical fixture contains 160,000,000 vertices and 400,000,000 edges. Both
sources independently generate and import the same seed-`20260829`, 64-shard
fixture, create all 18 range indexes, checkpoint cleanly, and restart in
read-only mode.

Run sequentially against four fresh databases:

1. Neo4j 4.4.48 uninterrupted clean migration with source before/after
   profiles and full canonical digest.
2. Neo4j 5.26.30 uninterrupted clean migration with source before/after
   profiles and full canonical digest.
3. Neo4j 4.4.48 recovery migration: SIGTERM near 25%, resume the same durable
   job, restart the Neo4j container near 40%, and resume again.
4. Neo4j 5.26.30 recovery migration: block PostgreSQL connectivity near 60%,
   restore and resume the same durable job, reboot the loader near 75%, and
   resume again.

Recovery runs use the clean source profiles as their immutability proof but
must recompute the complete target digest. Every resumed segment uses the same
database, graph, job ID, and frozen configuration fingerprint. No transient
service is treated as automatically resumed after a reboot.

## Acceptance and stopping

All gates in `acceptance.md` apply. In particular, each clean load must finish
within 24 hours, P3 post-warmup throughput must remain at least 70% of P2,
final/first phase throughput must remain at least 80%, p99 batch latency must
remain within three times its median, RSS must remain at most 2 GiB, and source
or target storage must stay below 80%.

Stop the current run without deleting evidence on any documented automatic
stop condition, if total live time reaches 72 hours, or if posted/projected
P3 cost reaches 225 USD. External governance stops are recorded as deviations;
the affected qualification run is resumed only after state and exclusivity
are proven.

After the final evidence is collected, deallocate all three VMs and stop the
Flexible Server. Keep every disk, database, Key Vault, fixture, log, and result
until cleanup receives separate authorization.
