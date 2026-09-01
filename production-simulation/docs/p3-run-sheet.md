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
fixture, create all 18 planner-usable source-key indexes (B-tree on Neo4j 4.4,
range on Neo4j 5.26), checkpoint cleanly, and restart in read-only mode. The
Neo4j 4.4 correction is required because that release can create range indexes
for migration preparation but cannot use them in Cypher query plans. Existing
range indexes and stopped-attempt evidence remain retained.

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

## Retained-job discovery recovery

The externally stopped Neo4j 4.4 clean run demonstrated that repeating live
discovery before every resume can itself exceed the 15-minute checkpoint-age
limit on a 560-million-element source. A reviewed discovery snapshot may
therefore replace only that repeated discovery step. It is derived from the
deterministic fixture headers and endpoint model, is version-bound by
`sourceId`, and must reproduce the retained iterator fingerprint exactly.
This does not change the frozen YAML, mapping queries, ordering, fetch size,
batch settings, source data, target database, graph, or durable job ID.

For a resumed P3 segment, pass the matching absolute snapshot path through
`P3_DISCOVERY_SNAPSHOT`. Preserve the original before-profile evidence, skip a
duplicate pre-resume full-source profile, and capture the after profile only
after the load commits. Any snapshot validation or checkpoint fingerprint
mismatch is an automatic stop with all existing evidence retained.

The retained Neo4j 4.4 r6 job then reached the first relationship mapping but
went more than 15 minutes without a committed checkpoint. Azure metrics showed
the source at approximately 99% CPU with effectively no data-disk reads,
consistent with the planner choosing an in-memory relationship scan/sort. The
run is retained as failed evidence. Generated relationship mappings must use
the portable `USING INDEX` hint for their source-key predicate; both Neo4j 4.4
B-tree and Neo4j 5.26 range indexes satisfy that hint. Because this changes the
source fingerprint, the corrected clean qualification starts in a fresh
database with a new durable job rather than weakening checkpoint validation or
resuming r6. When discovery proves that a relationship type has exactly one
primary endpoint-label pair, its generated query also omits the redundant
endpoint-label join. The reviewed Neo4j 4.4 plan is then a single ordered
`DirectedRelationshipIndexSeekByRange`, with no hash join or sort.

The corrected r7 run proved that large-to-small endpoint types meet the
throughput gate, but `CONTAINS` exposed a second source-side limit: fetching
random `external_id` properties from 45-million and 20-million-node endpoint
sets reduced throughput below the 24-hour projection gate. PostgreSQL used its
endpoint index (no identity-table sequential reads), while the Neo4j VM showed
low CPU and single-queue random disk reads. A bounded read-only profile returned
20,000 `id(a)`/`id(b)` endpoint pairs in 123 ms, approximately 285 times faster
than the measured property path. P3 sources are immutable and operationally
read-only, so subsequent fresh jobs explicitly use `vertexIdentity:
internal-id` for migration-time correlation. The visible `external_id` and all
other properties remain copied and are covered by the canonical digest.

The fresh r8 run validated the corrected fingerprint and reached 195,100,000
committed rows with zero rejected rows. It completed all vertices and the
35-million-edge `CARRIED_BY` mapping, but the first large-to-large `CONTAINS`
batches sustained only approximately 670--940 rows per second versus the
approximately 13,600 rows per second required by the 24-hour gate. The run was
therefore stopped with SIGTERM and retained as failed evidence. The Neo4j
internal-ID query no longer dominated the batch, but PostgreSQL had performed
70,203,183 endpoint identity index probes, approximately two per committed
edge. The next corrective investigation is consequently limited to target-side
endpoint resolution; r8 must not be resumed.

Azure reported automatic OS patch installations on two VMs at
2026-08-31T23:04:40Z. Loader and Neo4j boot times and service/container start
times were unchanged, the r8 checkpoint continued advancing, and no swap or
OOM event occurred. Record this platform action as a governance deviation, but
not as a restart or qualification fault injection.

The reviewed target-side correction retains the internal Neo4j identity in a
bounded, chunked dense cache for create/replace loads. Vertex mappings are made
visible to the cache only after the same target transaction commits; resume
rebuilds it from committed `vertex_identity` metadata. Label mismatches and
cache misses retain the existing PostgreSQL resolution path. Property identity
and incremental modes are unchanged. A five-run local benchmark resolved
20,000 edge endpoint pairs in 0.584--0.590 ms per batch, compared with the
approximately 21--30 seconds observed for the r8 target lookup. Qualification
still requires a fresh database and job plus measured Linux RSS and batch
throughput; r8 remains failed evidence and is not resumable for this change.
