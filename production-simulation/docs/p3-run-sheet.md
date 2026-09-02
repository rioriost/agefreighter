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
| Neo4j 4.4 source | `Standard_E8bds_v5`; 24 GiB heap / 28 GiB page cache | Frozen P2 source configuration |
| Neo4j 5.26 source | `Standard_E16bds_v5`; 48 GiB heap / 28 GiB page cache | User-authorized host and heap corrections after repeatable source OOM |
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

The first Neo4j 5.26 clean attempt stopped at 2026-09-02T00:30:00Z after the
source JVM reported `OutOfMemoryError` and the Bolt stream closed. The retained
job had committed 26,700,000 rows with zero rejects and a current checkpoint;
the loader used at most 589,840 KiB and did not swap. The Neo4j container was
not kernel-OOM-killed and did not restart, but its Java process remained at
approximately 52.5 GiB RSS after the exact source profile and initial load.
The attempt is failed clean-run evidence and must not be resumed.

The source is operationally read-only, and the failed attempt did not mutate
it. A fresh full P3 retry may therefore reuse the retained exact source-before
profile only when its JSON is valid, its hash and original absolute path are
recorded, the source version/read-only state and disk are reproven after a
container restart, and a fresh target database and durable job are used. This
avoids immediately repeating the full source scan that preceded the JVM memory
failure without changing the frozen Neo4j heap/page-cache settings, migration
configuration, source data, or post-load source-profile requirement.

After the source container stopped, host available memory returned above 61
GiB. It restarted as Neo4j Community 5.26.30 with the database online and
read-only, 32% source-disk use, no swap, and no kernel OOM. Fresh run
`clean-r11-neo4j526` created isolated PostgreSQL 18 / AGE 1.7 targets and
started at 2026-09-02T01:41:54Z. It retained the source-before profile with
SHA-256 `0b7b16a969f17ac5843f93b49999d00cc462985319d7db8c6be3ab30f2b9c900`
and began a new uninterrupted load using the unchanged frozen configuration.

Fresh r11 reproduced the same failure at 2026-09-02T02:18:09Z: its source JVM
reported `OutOfMemoryError` after exactly 26,700,000 committed rows, with zero
rejects, while the loader remained below 570 MiB RSS and did not swap. The
source Java process again reached approximately 52.5 GiB RSS on the 64 GiB VM;
the container was neither restarted nor kernel-OOM-killed. This rules out the
preceding standalone source profile as the cause and establishes a repeatable
source-memory limit with the frozen 24 GiB heap / 28 GiB page cache on
`Standard_E8bds_v5`. The r11 job is retained and must not be resumed.

The source container was stopped after evidence collection, all three VMs were
deallocated, and the Flexible Server was stopped. Do not attempt another clean
run with the unchanged source host. The least invasive corrective option is to
resize only the Neo4j 5.26 source VM to a 128 GiB class while keeping its heap,
page cache, disk, source data, and migration configuration unchanged, but this
changes a frozen P3 infrastructure value and requires explicit authorization.

The user authorized that correction on 2026-09-02. The deallocated Neo4j 5.26
VM was resized from `Standard_E8bds_v5` to `Standard_E16bds_v5` (128 GiB) in
zone 1. Its 24 GiB heap, 28 GiB page cache, Premium SSD v2 data disk, immutable
source, and all migration settings remain unchanged. A new clean attempt must
use another fresh target database and durable job; r10 and r11 remain failed
evidence and are never resumed.

The resized guest then proved 16 vCPUs and approximately 126 GiB usable
memory. Neo4j Community 5.26.30 was online and read-only with no swap or kernel
OOM event. PostgreSQL 18.6 was `Ready` / HA `Healthy` and accepted an
authenticated private-network connection; the loader had no competing P3
unit. Fresh run `clean-r12-neo4j526` started at 2026-09-02T03:38:23Z on source
commit `b64382da9d5545d57edf10f961fe9e035e26347e`, using a fresh target and the
retained validated source-before profile with SHA-256
`0b7b16a969f17ac5843f93b49999d00cc462985319d7db8c6be3ab30f2b9c900`. Keep
r12 uninterrupted, record its new durable job ID when created, and confirm it
progresses beyond both retained 26.7-million-row failure boundaries before
advancing the qualification state.

R12 created durable job `2044af71-10b7-4a7e-84a3-31f2702ca42b` but failed at
2026-09-02T04:14:12Z after exactly 26,700,000 committed rows and zero rejects.
The loader maximum RSS was 587,640 KiB with zero swap. The expanded source host
had approximately 126 GiB usable memory, while the unchanged Neo4j process
retained its fixed 24 GiB Java heap and 28 GiB page cache. Repeated full-GC
pauses preceded JVM `OutOfMemoryError` events. Java RSS was approximately 52.6
GiB; neither the kernel nor the container cgroup recorded an OOM kill, and the
container did not restart. This third identical boundary proves that host RAM
alone does not enlarge the fixed JVM heap.

R12 is failed evidence and must not be resumed. Its source container stopped
cleanly at 2026-09-02T04:55:28Z; all three VMs are confirmed deallocated and
the Flexible Server is confirmed stopped. A 48 GiB heap with the page cache
held at 28 GiB is the minimum recommended correction on the existing 128 GiB
source VM, leaving substantial memory for the OS and file cache. Because this
changes a second frozen P3 source value, do not start r13 until the user
explicitly authorizes that heap correction.

The user explicitly authorized the 48 GiB Neo4j 5.26 heap correction on
2026-09-02. Keep the page cache at 28 GiB, the source VM at
`Standard_E16bds_v5`, and all source data, disk, image, migration, loader, and
target settings unchanged. Preserve r12 and its stopped-container evidence,
reprove the new effective heap and read-only database state, and use a fresh
r13 target database and durable job. R13 must still be uninterrupted and must
progress beyond the three retained 26.7-million-row failure boundaries before
the clean qualification can advance.

The stopped r12 container and its logs were retained separately. The
replacement source container proved the pinned Neo4j 5.26.30 image, 48 GiB
initial/maximum heap, 28 GiB page cache, online read-only database access, 32%
source-disk use, and zero swap or kernel OOM events. After authenticated private
PostgreSQL connectivity returned, fresh run `clean-r13-neo4j526` started at
2026-09-02T06:10:20Z on commit
`8d2e61ca43adb12a510ab63707702115be899c0b`. It prepared another fresh target
and reused the validated source-before profile with SHA-256
`0b7b16a969f17ac5843f93b49999d00cc462985319d7db8c6be3ab30f2b9c900`. Record
the new durable job ID when created and monitor it uninterrupted past the three
retained failure boundaries.

R13 created durable job `0edba78e-6eff-4fd7-9493-f01a3298546d`. At
2026-09-02T06:48:05Z it had committed 37,160,000 rows with zero rejects, next
batch 1,859, and a current checkpoint. This passes the exact 26.7-million-row
boundary shared by r10, r11, and r12. Loader RSS was 729,364 KiB, while source
Java RSS was 80,827,544 KiB with approximately 50.4 GB host memory available.
Source and loader swap, kernel OOM, Java OOM, and container restarts remained
zero. Keep r13 uninterrupted and continue monitoring all P3 guardrails through
the 560-million-row commit and complete verification sequence.

R13 then failed at 2026-09-02T06:56:29Z after 47,540,000 committed rows and
zero rejects. Its durable resume token records vertex mapping 4 (`Shipment`),
41,540,000 rows into that mapping. The loader maximum RSS was 907,872 KiB with
zero swap. Source Java RSS reached approximately 80.9 million KiB and repeated
full-GC pauses preceded `OutOfMemoryError`; the kernel and container cgroup did
not OOM-kill the process, and the container did not restart. Increasing the
heap from 24 GiB to 48 GiB moved the failure from approximately 20.7 million to
41.54 million rows inside the same generated query. This establishes that one
long-lived Bolt query retains server-side state proportional to consumed rows.

Retain r13 and never resume it. Its stopped container and logs have a guest-side
checksum. The corrective query plan keeps the existing unique-key ordering but
adds `LIMIT $pageRows`; the iterator closes each result and resumes from the
last unique key. With the frozen `fetchRows: 5000`, this bounds every
server-side auto-commit query to 5,000 rows instead of one complete mapping.
The full Go test suite passes. The 48 GiB heap, 28 GiB page cache, source image
and data, 8-byte internal identity, migration limits, and target settings remain
unchanged. Use a fresh r14 database and durable job because the generated-query
fingerprint intentionally changes.

Fresh run `clean-r14-neo4j526` started at 2026-09-02T08:11:31Z on commit
`9391b23c7527ea35338fa731eb18a88136efaa65`. The replacement source proved a
48 GiB initial/maximum heap, 28 GiB page cache, online read-only database,
retained r12/r13 containers, and zero swap/OOM events. PostgreSQL 18.6 was
`Ready` / HA `Healthy`; loader preflight proved 41% disk use, no competing P3
unit, and the unchanged validated source-before profile. R14 prepared a fresh
target and began the corrected uninterrupted load. Record its durable job ID
when created, then compare source heap behavior and checkpoint progress through
both retained failure boundaries before advancing.

R14 created durable job `01e74add-b171-4cc7-96ad-44a5d1035ffc`. At
2026-09-02T08:49:23Z it had committed 41,420,000 rows with zero rejects and a
current checkpoint, passing the 26.7-million-row boundary shared by r10-r12.
Loader RSS remained below 1 GiB and the source retained approximately 50.5 GB
of available host memory. Swap, kernel/Java OOM events, and container restarts
were zero; loader and source data-disk use remained 41% and 32%. Continue the
uninterrupted run through r13's 47.54-million-row boundary before advancing.

At 2026-09-02T09:54:13Z r14 had committed 131,940,000 rows with zero rejects,
next batch 6,598, and a 342-second checkpoint age. It has passed both retained
failure boundaries and reached almost three times r13's terminal row count.
Loader RSS was 1,677,352 KiB, source Java RSS was 80,794,316 KiB, and the
source host retained approximately 50.4 GB available memory. Swap, OOM events,
and container restarts remained zero. Flexible Server storage was 39.21%, the
posted P3 actual cost was 290.22 USD against the 800 USD ceiling, PostgreSQL
was `Ready` / HA `Healthy`, and no external governance stop occurred. Keep r14
uninterrupted through completion and then run the full clean verification.

At 2026-09-02T11:50:54Z r14 had committed 299,860,000 of 560,000,000 rows
(53.5%) with zero rejects and a current checkpoint. Loader RSS was 2,732,008
KiB, source Java RSS was 80,917,596 KiB, and swap, OOM events, and container
restarts remained zero. Azure installed OS updates on both active VMs between
11:04:40Z and 11:08:00Z, but neither guest rebooted, the loader unit and source
container retained their original start times, and checkpointed progress
continued. Record this as a non-interrupting external action. Flexible Server
storage was 42.06%. The cost endpoint was rate-limited, so the latest posted
actual remains 290.22 USD against the 800 USD ceiling. Continue r14 without
interruption.

R14 committed all 560,000,000 rows with zero rejects and 28,000 completed
batches at 2026-09-02T13:48:53Z. The uninterrupted load completed in 5:37:01
with exit status zero and maximum loader RSS 2,733,644 KiB, within the 4 GiB
gate. The source remained online and read-only with no swap, OOM event, or
container restart. The service then advanced directly to the exact job report
and count collection, which remained active at 13:53:18Z. Flexible Server
storage was 46.40%, within the 80% gate. Do not interrupt the post-load
sequence; complete the remaining checks and full canonical digest before
qualifying the Neo4j 5.26 clean run.

By 2026-09-02T14:49:21Z the exact job report, built-in verification, exact
source-after profile, doctor, pre/post-`ANALYZE` optimization reviews, target
`ANALYZE`, and deterministic fixture digest had all completed successfully.
The full 5,600-range target canonical digest then started and remained active
at 14:51:20Z. Source Java RSS was stable at 81,028,300 KiB with approximately
50.2 GB host memory available; swap, OOM events, and container restarts were
zero. Flexible Server storage was 46.35%. The cost endpoint remained
rate-limited and the latest posted actual remains 290.22 USD against the 800
USD ceiling. Do not interrupt the target digest; compare its complete output
with the fixture digest before accepting the clean run.

At 2026-09-02T15:06:49Z the external Azure governance principal stopped the
Flexible Server while the first r14 target-digest attempt was active, then
deallocated the loader and Neo4j 5.26 VMs at 15:47Z. The digest emitted no
target output. This was not an authorized qualification fault and did not
change the committed job or completed clean-run evidence. The posted P3 actual
cost was 353.61 USD within the 800 USD ceiling; the last storage reading was
46.35%, within the 80% gate.

Restart only the loader and Flexible Server and keep both source VMs
deallocated. After the private-network startup gate cleared, PostgreSQL was
`Ready` / HA `Healthy`; the retained job again proved `committed` with
560,000,000 rows and zero rejects, no P3 unit was active, loader disk use was
41%, and swap/OOM was zero. Checksums for the core r14 evidence were recorded,
and the interrupted attempt had emitted no partial target digest. Retained-job
digest retry `clean-r14-digest-r2-neo4j526` started at
2026-09-02T15:58:49Z against verifier commit
`9391b23c7527ea35338fa731eb18a88136efaa65` and was active at 15:59:40Z.
Preserve the interrupted r14 service evidence and let r2 finish uninterrupted.
