# P3 reviewed run sheet

## Authorization and deadline

The user authorized P3 execution and its forecast budget on 2026-08-31. P3
must complete during the week ending 2026-09-06. The reviewed live-operation
window was initially 72 hours and was extended to 96 hours on 2026-09-02 so
monitoring can continue without requiring the user to check this task at a
specific time. On 2026-09-01 the user raised the cost ceiling first to
400 USD and then to 800 USD, and made completing all four qualification runs
the highest priority. This authorization covers deployment, source
preparation, four migration runs, the
controlled faults below, full verification, monitoring, evidence retention,
and stopping the resources. It does not authorize P4 or deletion of resources
or evidence.

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
`batchRows: 20000`, `batchBytes: 64MiB`, a 4 GiB loader memory limit, and
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
remain within three times its median, RSS must remain at most 4 GiB, and source
or target storage must stay below 80%.

Stop the current run without deleting evidence on any documented automatic
stop condition, if total live time reaches 96 hours, or if posted/projected
P3 cost reaches 800 USD. External governance stops are recorded as deviations;
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

Azure repeated the automatic patch action on the loader and Neo4j 4.4 VMs from
2026-09-01T05:04:40Z through 2026-09-01T05:08:01Z during r10, then again from
2026-09-01T11:04:40Z through 2026-09-01T11:08:01Z during its digest retry.
Guest evidence showed package assessment/update activity, including
`ubuntu-advantage-tools`, but no VM, loader process, or Neo4j container restart.
The r10 checkpoint and later digest continued advancing with no swap or OOM
event. Retain these actions as governance deviations, not as qualification
fault injections.

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

The fresh r9 run reached the cached edge stage and 221,620,000 committed rows
with zero rejected rows and a current checkpoint, proving that endpoint lookup
no longer stalls the batch. Its loader RSS nevertheless reached approximately
2.61 GiB, above the then-current 2 GiB acceptance gate, so r9 was stopped with
SIGTERM and retained as failed evidence. Review retained the generic 8-byte AGE
graph ID representation and raised only the P3 loader memory/RSS limit to
4 GiB; earlier phases and release gates remain at 2 GiB. The configuration and
job fingerprint therefore change, requiring another fresh database and job;
r9 must not be resumed.

The r10 Neo4j 4.4 clean load then committed all 560,000,000 records with zero
rejects in 3:55:27 and a 2.61 GiB peak RSS. Its first target-digest invocation
stopped before emitting target output because the independent verifier assumed
that `vertex_identity.external_id` was always the visible `external_id`
property. P3 deliberately stores the Neo4j internal ID there for migration-time
correlation while retaining the visible property in AGE. The verifier must
therefore read canonical vertex and endpoint identities from the referenced AGE
properties, while retaining metadata only to select and join the committed
generation. Retry only the digest against the same immutable committed r10 job;
do not rerun or alter the qualified load.

Digest retry r2 corrected identity semantics but its target query joined both
endpoint vertex tables and allowed PostgreSQL to sort the expanded relation.
After 3:44:17 PostgreSQL exhausted its temporary tablespace and aborted before
emitting target output; the committed graph and expected digest remained
unchanged, and server storage stayed near 34%. The next verifier retains a
bounded, chunked 8-byte GraphID-to-source-key index while digesting vertices,
then resolves edge endpoints locally and removes both endpoint joins. Its
session disables parallel, hash, merge, and explicit sort plans. Reviewed live
`EXPLAIN` output uses the generation/label/GraphID identity indexes as the outer
stream and each AGE label primary key as the inner lookup, with no sort node.
Retry r3 only against the same immutable committed r10 job.

At 2026-09-01T15:10:38Z the external Azure governance principal stopped the
Flexible Server while retry r3 was reading the committed graph. The verifier
failed with a target connection timeout at 2026-09-01T15:11:44Z and emitted no
target digest. This was not an authorized qualification fault and did not
change the committed job. The server was restarted, returned to `Ready` with
SameZone HA `Healthy` at 2026-09-01T15:30:59Z, and the retained job was proven
`committed` with 560,000,000 committed rows, zero rejects, and no other active
load job. Retry r4 started at 2026-09-01T15:33:21Z against that same immutable
database and job; r3 remains retained as a governance-deviation failure.

The same external governance principal deallocated the loader and Neo4j 4.4
VMs at 2026-09-01T15:46Z while retry r4 was running. The abrupt loader stop
terminated r4 before it emitted a target digest; its partial evidence is
retained. The attached disks were preserved, and the source data disk still
reports the frozen 10,000 IOPS / 750 MB/s configuration. After restarting only
the loader, guest checks showed 41% OS-disk use, no swap or OOM event, and no
active P3 unit. PostgreSQL remained `Ready` / HA `Healthy`, and the same target
job was again proven `committed` with 560,000,000 rows and zero rejects. Retry
r5 started at 2026-09-01T16:27:24Z; both externally interrupted retries remain
retained and are not reused.

At 2026-09-01T17:24Z Azure posted P3 cost of 396.08 USD. Continuing the active
digest through the next observation would project beyond the authorized 400
USD ceiling, so retry r5 received `SIGTERM` at 2026-09-01T17:25:35Z before it
emitted target output. The loader was deallocated and the Flexible Server was
stopped; all data and partial evidence remain retained. A fourth Azure
patch action completed on the restarted loader from 2026-09-01T17:04:40Z to
17:13:00Z without changing its boot time, and no swap or OOM event occurred.
P3 was paused at the cost gate pending revised budget authorization.

Later on 2026-09-01 the user authorized an 800 USD ceiling and requested that
P3 resume. Azure's posted actual-cost result had also been revised from 396.08
USD to 253.00 USD. The loader and Flexible Server were restarted while both
source VMs remained deallocated. After the transient private-network startup
window cleared, PostgreSQL was `Ready` / HA `Healthy`; the retained job again
proved `committed` with 560,000,000 rows, zero rejects, and no competing active
job. Loader disk use was 41%, with no swap or OOM event. Digest retry r6
started at 2026-09-01T20:32:48Z against the same immutable database and job.

Retry r6 completed at 2026-09-01T22:46:21Z after 2:13:21. It produced all
5,600 target leaves for 560,000,000 rows, and every range plus the final root
matched the retained expected digest
`0302c456d17c6e9ee64552d68e2bf6a775e63cd3b09120f5bc342d329bddd1ba`.
Maximum verifier RSS was 2,633,072 KiB, with zero swaps and exit status 0. A
fifth automatic patch window completed on the loader from
2026-09-01T23:04:40Z to 23:08:00Z without changing its boot time or disturbing
the completed evidence. The Neo4j 4.4 clean qualification is complete; advance
to the Neo4j 5.26 clean source preflight.

The retained Neo4j 5.26 source then proved Community 5.26.30, zone 1,
read-only database access, 32% source-disk use, and no competing P3 process.
The untouched paired r10 target contained AGE 1.7.0 with no migration schema or
graph. Run `clean-r10-neo4j526` completed its exact source-before profile and
started the uninterrupted load at 2026-09-01T23:53:05Z with durable job
`3e8f78bb-fea3-4cd6-b726-77652d95d709`. At the first live checkpoint it had
committed 26,680,000 rows with zero rejects, a 23-second checkpoint age, and
approximately 537 MiB RSS. A sixth automatic patch window completed on the
source VM from 2026-09-01T23:43:40Z to 23:53:00Z without restarting the VM or
Neo4j container; swap and OOM remained zero.
