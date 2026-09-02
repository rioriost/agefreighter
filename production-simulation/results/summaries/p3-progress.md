# P3 production-simulation progress

This is the redacted live progress report for the P3 qualification in
`rg-afps-p3-20260831`. It is updated at material phase transitions; retained
guest evidence and the final reviewed result remain authoritative.

- Updated: 2026-09-02T20:59:31Z
- Overall state: **RUNNING**
- Current position: **Neo4j 4.4 recovery — segment 2 resumed after the planned 25% fault**
- Next: restart the Neo4j 4.4 container near 224,000,000 committed rows
- Target: PostgreSQL 18 / Apache AGE 1.7 on Azure Database for PostgreSQL
  Flexible Server

## Qualification overview

| Source | Qualification run | State | Current or next step |
| --- | --- | --- | --- |
| Neo4j 4.4.48 | Clean | **DONE** | All 560,000,000 rows and 5,600 digest ranges match |
| Neo4j 4.4.48 | Recovery | **RUNNING** | Planned `SIGTERM` retained at 140,920,000 rows; segment 2 resumed toward the ~224,000,000-row source restart |
| Neo4j 5.26.30 | Clean | **DONE** | All 560,000,000 rows and 5,600 digest ranges match |
| Neo4j 5.26.30 | Recovery | PENDING | Start after both clean-source qualifications |

## Neo4j 4.4.48

### Clean qualification

| Step | Evidence gate | State | Current evidence |
| ---: | --- | --- | --- |
| 1 | Target preflight, plan, and exact source profile before load | DONE | Source profile and plan retained |
| 2 | Uninterrupted 560-million-record migration | DONE | 560,000,000 committed in 3:55:27; zero rejects and failed batches |
| 3 | Job report and exact count collection | DONE | `report.json` retained |
| 4 | Built-in counts, catalog, generation, and bounded integrity checks | DONE | All executed checks pass; bounded coverage remains `incomplete` until the full digest closes it |
| 5 | Exact source profile after load | DONE | Before/after profiles retained for immutability review |
| 6 | Doctor and pre-`ANALYZE` optimization review | DONE | Evidence retained |
| 7 | Target `ANALYZE` and post-`ANALYZE` optimization review | DONE | `ANALYZE` and post-analysis review completed at 2026-09-01T06:50:16Z |
| 8 | Deterministic fixture manifest and expected range digest | DONE | 5,600 expected leaves and fixture root retained |
| 9 | Full target canonical range digest | DONE | Retry r6 emitted 560,000,000 rows and 5,600 leaves in 2:13:21; maximum RSS 2,633,072 KiB |
| 10 | Range/root comparison and run summary | DONE | Every range and final root match the retained expected digest |

Clean-run identifiers and measured gates:

- Run: `clean-r10-neo4j44`
- Durable job: `960a6813-4710-48c5-a09f-a08c637b6aeb`
- Load result: 560,000,000 read and committed; zero rejects; 28,000 committed
  batches
- Peak loader RSS: 2,741,392 KiB (2.61 GiB), below the 4 GiB P3 gate
- Swap/OOM: none

The first target-digest invocation stopped at 2026-09-01T08:20:30Z before
producing target digest output. Its verifier assumed that the operational
`vertex_identity.external_id` always equaled the visible `external_id`
property. P3 intentionally uses Neo4j internal IDs for migration-time endpoint
correlation while preserving the visible property, so that assumption was not
valid. The committed graph, expected fixture digest, and all earlier evidence
remain unchanged. The correction derives canonical vertex and endpoint
identities from the referenced AGE properties and retries only the independent
target digest against the same committed job.

The first detached retry launcher retained an immediate pre-verifier bootstrap
failure because its transient service did not define the Go module/cache
environment. No database command ran. The corrected launcher uses explicit
root-owned build caches and a new evidence directory; the failed retry remains
retained and is not reused. Retry r2 started the corrected target digest at
2026-09-01T09:38:19Z against the same immutable committed job.

Retry r2 corrected identity semantics but PostgreSQL chose a target-side sort
for its endpoint-expanded query. The query stopped at 2026-09-01T13:22:36Z
after 3:44:17 when the temporary tablespace filled; no target digest was
emitted and the committed graph was not changed. Retry r3 builds a bounded,
chunked 8-byte GraphID-to-source-key index while digesting vertices, resolves
edge endpoint identities locally, and removes the two per-edge endpoint joins.
Live `EXPLAIN` evidence shows index-ordered nested loops with no sort node.
Retry r3 started at 2026-09-01T14:27:39Z against the same immutable committed
job.

At 2026-09-01T15:10:38Z an external Azure governance action stopped the
Flexible Server. Retry r3 then failed with a target connection timeout at
2026-09-01T15:11:44Z and emitted no digest. The server was restarted and
returned to `Ready` / SameZone HA `Healthy` at 2026-09-01T15:30:59Z. Before
retrying, the retained job was proven `committed` with 560,000,000 committed
rows, zero rejects, next batch 28,001, and no other active load job. Retry r4
started at 2026-09-01T15:33:21Z against that unchanged database and job. Its
verifier process was active with no swap or OOM event; retry r3 remains retained
as a governance-deviation failure.

At 2026-09-01T15:46Z the same external governance principal deallocated both
the loader and Neo4j 4.4 VMs. The loader shutdown terminated retry r4 before it
emitted a target digest. Its evidence is retained, the attached disks remain
intact, and the source data disk still has the frozen 10,000 IOPS / 750 MB/s
configuration. The loader was restarted with 41% OS-disk use, no swap or OOM
event, and no other active P3 unit. PostgreSQL remained `Ready` / HA `Healthy`,
and the immutable job was again proven `committed` with 560,000,000 rows and
zero rejects. Retry r5 started at 2026-09-01T16:27:24Z and is active. The Neo4j
4.4 VM remains deallocated until its next required qualification step.

Retry r5 ran from 2026-09-01T16:27:24Z until the cost gate at
2026-09-01T17:25:35Z. At the last live observation its verifier had 2,096,600
KiB RSS, read approximately 104.5 GB from the loader filesystem, and had no
swap or OOM event. Azure then posted 396.08 USD. Because continuing through the
next observation projected beyond the authorized 400 USD ceiling, r5 received
`SIGTERM` before emitting target output. All three VMs were deallocated and the
Flexible Server was stopped, with all disks and evidence retained. A
fourth automatic patch window completed at 2026-09-01T17:13:00Z without
rebooting the loader. Further P3 execution requires revised budget
authorization.

The user then authorized an 800 USD ceiling and requested resumption. Azure's
posted actual-cost value was revised from 396.08 USD to 253.00 USD. The loader
and Flexible Server were restarted while both source VMs remained deallocated.
After the transient private-network startup window cleared, PostgreSQL was
`Ready` / HA `Healthy`; the same job again proved `committed` with 560,000,000
rows, zero rejects, next batch 28,001, and no competing active job. The loader
had 41% disk use, no swap, and no OOM event. Retry r6 started at
2026-09-01T20:32:48Z and is active against the unchanged database and job.

Retry r6 completed at 2026-09-01T22:46:21Z with exit status 0. The target
digest covered all 560,000,000 rows in 5,600 leaves. Every range and the final
root matched the retained expected root
`0302c456d17c6e9ee64552d68e2bf6a775e63cd3b09120f5bc342d329bddd1ba`.
Maximum verifier RSS was 2,633,072 KiB with zero swaps. A fifth automatic patch
window completed at 2026-09-01T23:08:00Z without rebooting the loader or
altering the completed evidence. The clean qualification is complete.

### Recovery qualification

| Step | Fault/recovery evidence | State |
| ---: | --- | --- |
| 1 | Start a fresh target database and durable job | DONE; job `d727d9aa-b7e5-4728-9254-90bd0210406d` |
| 2 | Send loader `SIGTERM` near 25% and retain the checkpoint | DONE; `SIGTERM` at 140,920,000 rows (25.16%) with a current checkpoint |
| 3 | Resume the same database, graph, job, and fingerprint | DONE; segment 2 resumed job `d727d9aa-b7e5-4728-9254-90bd0210406d` with the retained fingerprint |
| 4 | Restart the Neo4j 4.4 container near 40% | **RUNNING**; monitor segment 2 toward 224,000,000 rows |
| 5 | Resume the same durable job again and finish the load | PENDING |
| 6 | Run the complete built-in post-load checks | PENDING |
| 7 | Compute the complete target digest and match fixture and clean roots | PENDING |

The recovery run reuses the accepted clean source profiles as the source
immutability proof, but it must compute a new complete target digest.

The cold live-discovery phase completed and created durable job
`d727d9aa-b7e5-4728-9254-90bd0210406d`. At 2026-09-02T20:12:49Z it had
committed 31,000,000 rows with zero rejects, next batch 1,551, and a current
checkpoint. Loader RSS was 659,156 KiB, within the 4 GiB gate. Source Java RSS
was 55,382,128 KiB with approximately 9.4 GB host memory available; loader and
source swap/OOM remained zero, the source container had not restarted, and its
data disk remained 32% used. Flexible Server storage was 46.82%, within the
80% gate. The cost endpoint was rate-limited; the latest posted actual remains
353.61 USD against the 800 USD ceiling. Keep segment 1 active until the planned
`SIGTERM` near 140,000,000 committed rows.

After a final pre-fault guardrail check, segment 1 received `SIGTERM` at
2026-09-02T20:56:02Z with 140,920,000 committed rows, zero rejects, next batch
7,047, and a current checkpoint. Loader RSS was 2,424,200 KiB, disk use was
41%, and swap/OOM were zero. The process exited, the durable job retained its
checkpoint, and the fault evidence plus SHA-256 manifest is stored under the
segment 1 result directory. Flexible Server was `Ready` / HA `Healthy` with
48.85% storage immediately before the fault; the cost endpoint remained
rate-limited and the latest posted actual was 353.61 USD.

Segment 2 started at 2026-09-02T20:58:45Z against the same database, graph,
generation, durable job, and configuration fingerprint
`7ce3e588f60dae063069347b77fa17fdf0db17981be259df8911454f7a45876b`.
It uses the reviewed Neo4j 4.4 discovery snapshot at verifier commit
`9391b23c7527ea35338fa731eb18a88136efaa65`; the snapshot SHA-256 is
`1cc823addfe580f3cb9e5a6c2bb315a49568bb966d38bec5aa3497b19d3e4036`.
At 20:59:31Z the resume process was active with the same 140,920,000-row
checkpoint and zero rejects. Continue uninterrupted to the planned Neo4j
container restart near 224,000,000 committed rows.

## Neo4j 5.26.30

### Clean qualification

| Step | Evidence gate | State |
| ---: | --- | --- |
| 1 | Power on the retained source VM and prove version, zone, read-only state, and exclusivity | DONE |
| 2 | Target preflight, plan, and exact source profile before load | DONE; retained for fresh retry |
| 3 | Uninterrupted 560-million-record migration | DONE; r14 committed all rows in 5:37:01 |
| 4 | Job report and exact count collection | DONE |
| 5 | Built-in counts, catalog, generation, and bounded integrity checks | DONE |
| 6 | Exact source profile after load | DONE |
| 7 | Doctor, optimization review, target `ANALYZE`, and post-`ANALYZE` review | DONE |
| 8 | Deterministic fixture manifest and expected range digest | DONE |
| 9 | Full target canonical range digest | DONE; 560,000,000 rows in 5,600 leaves |
| 10 | Range/root comparison and run summary | DONE; every range and root match |

The retained source is Neo4j Community 5.26.30 in Japan East zone 1. Its data
disk is 32% used, configured at 10,000 IOPS / 750 MB/s, and the `neo4j`
database is online and read-only. No swap, OOM event, or competing P3 unit was
present. The untouched target `agefreighter_p3_clean_r10_neo4j526` contains
AGE 1.7.0 with no migration schema or graph. Clean run
`clean-r10-neo4j526` started at 2026-09-01T23:33:16Z using verifier/source
commit `2bc32cd3366fee39062258de05772945b08c0607`; validation and target preflight
passed, the plan was retained, and the exact source-before profile began.

The exact source-before profile completed at 2026-09-01T23:53:05Z and the
uninterrupted load started with durable job
`3e8f78bb-fea3-4cd6-b726-77652d95d709`. At 2026-09-02T00:27:25Z it had
committed 26,680,000 rows with zero rejects, next batch 1,335, and a 23-second
checkpoint age. Loader RSS was approximately 537 MiB. PostgreSQL storage was
34.50%, source storage was 32%, and swap/OOM remained zero. The automatic patch
window from 2026-09-01T23:43:40Z to 23:53:00Z did not restart the source VM or
Neo4j container.

At 2026-09-02T00:30:00Z the Bolt stream closed after the source JVM reported
`OutOfMemoryError`. The retained failed job had committed 26,700,000 rows with
zero rejects and a current checkpoint; loader maximum RSS was 589,840 KiB with
no swap. The Neo4j container was not kernel-OOM-killed and had not restarted,
but its Java process remained at approximately 52.5 GiB RSS. The container was
stopped after evidence collection and the failed clean job will not be
resumed. The next attempt uses a fresh target database and durable job after a
source restart. Because the source is read-only and unchanged, the validated
source-before profile is reused with its original path and SHA-256 recorded;
the frozen source and migration configuration are unchanged.

The source container restarted as Neo4j Community 5.26.30 with the database
online and read-only, 32% source-disk use, no swap, and no kernel OOM. Fresh
run `clean-r11-neo4j526` created isolated PostgreSQL 18 / AGE 1.7 targets and
started at 2026-09-02T01:41:54Z. It retained the source-before profile with
SHA-256 `0b7b16a969f17ac5843f93b49999d00cc462985319d7db8c6be3ab30f2b9c900`
and is initializing a new uninterrupted load with the frozen configuration.

Fresh r11 reproduced the same source-JVM `OutOfMemoryError` at
2026-09-02T02:18:09Z after exactly 26,700,000 committed rows and zero rejects.
The loader remained below 570 MiB RSS with no swap. The source Java process
again reached approximately 52.5 GiB RSS; the container was not restarted or
kernel-OOM-killed. Reproducing the same boundary after a source restart rules
out the preceding standalone profile as the cause and establishes a source VM
memory limit for the frozen 24 GiB heap / 28 GiB page cache on the 64 GiB VM.
The failed r11 job will not be resumed. The source container was stopped, all
VMs were deallocated, and PostgreSQL was stopped with all evidence retained.
Resizing only the Neo4j 5.26 source VM to a 128 GiB class is the least invasive
correction, but requires authorization because VM size is a frozen P3 value.

The user authorized resizing only the Neo4j 5.26 source VM. While deallocated,
it was changed from `Standard_E8bds_v5` to `Standard_E16bds_v5` (128 GiB) in
zone 1. The 24 GiB heap, 28 GiB page cache, source disk and data, and migration
configuration remain unchanged. The next clean attempt uses another fresh
target database and durable job; r10 and r11 remain retained failed evidence.

After the resize, the guest proved 16 vCPUs and approximately 126 GiB usable
memory. Neo4j Community 5.26.30 returned online and read-only with 32% source
disk use, no swap, and no kernel OOM event. PostgreSQL 18.6 returned to
`Ready` / HA `Healthy` and accepted an authenticated private-network
connection. The loader had 41% disk use, no swap or OOM event, and no competing
P3 unit. Fresh run `clean-r12-neo4j526` started at 2026-09-02T03:38:23Z on
commit `b64382da9d5545d57edf10f961fe9e035e26347e`, prepared a new PostgreSQL 18 /
AGE 1.7 target, and reused the retained source-before profile with SHA-256
`0b7b16a969f17ac5843f93b49999d00cc462985319d7db8c6be3ab30f2b9c900`.
At the startup phase transition the uninterrupted load process was active and
its durable job row had not yet been created. The prior r10 and r11 jobs remain
retained and are not resumed.

Fresh r12 created durable job `2044af71-10b7-4a7e-84a3-31f2702ca42b` and then
failed at 2026-09-02T04:14:12Z after exactly 26,700,000 committed rows and zero
rejects, reproducing the r10 and r11 boundary. The loader used at most 587,640
KiB RSS with zero swap. The expanded host had approximately 126 GiB usable
memory, but the unchanged Neo4j process still had a fixed 24 GiB Java heap and
reported repeated `OutOfMemoryError` events after sustained full-GC pauses.
Java RSS was approximately 52.6 GiB. The container was not kernel- or
cgroup-OOM-killed, did not restart, and its data disk remained 32% used. This
proves that increasing host RAM alone cannot correct the fixed JVM-heap limit.

The failed r12 target and guest evidence are retained and must not be resumed.
The source container was stopped cleanly at 2026-09-02T04:55:28Z. All three
VMs are now confirmed deallocated and the Flexible Server is confirmed
stopped. The recommended minimum correction is a 48 GiB Neo4j heap with the
page cache held at 28 GiB on the existing 128 GiB VM. This changes another
frozen P3 source value and requires explicit authorization before a fresh r13
run.

The user explicitly authorized that correction on 2026-09-02. The tracked P3
Neo4j 5.26 configuration now uses a 48 GiB initial/maximum heap while retaining
the 28 GiB page cache. The 128 GiB VM, source image and data, disk settings,
migration configuration, loader, and PostgreSQL target remain unchanged. R13
will preserve the stopped r12 container evidence, reprove effective memory and
read-only state, and use another fresh database and durable job.

The stopped r12 container and its logs were retained separately. The new
source container proved the pinned Neo4j 5.26.30 image, a 48 GiB
initial/maximum heap, a 28 GiB page cache, online read-only database access,
32% source-disk use, and zero swap or kernel OOM events. Fresh run
`clean-r13-neo4j526` started at 2026-09-02T06:10:20Z on commit
`8d2e61ca43adb12a510ab63707702115be899c0b`, prepared a new PostgreSQL 18 /
AGE 1.7 target, and reused the retained exact source-before profile with SHA-256
`0b7b16a969f17ac5843f93b49999d00cc462985319d7db8c6be3ab30f2b9c900`.
The load process is active with no swap/OOM event; its durable job row has not
yet been created.

R13 created durable job `0edba78e-6eff-4fd7-9493-f01a3298546d`. At
2026-09-02T06:48:05Z it had committed 37,160,000 rows with zero rejects, next
batch 1,859, and a current checkpoint. It has therefore passed the exact
26.7-million-row boundary shared by r10, r11, and r12. Loader RSS was 729,364
KiB, below the 4 GiB gate. Source Java RSS was 80,827,544 KiB with approximately
50.4 GB host memory available; source and loader swap, kernel OOM, Java OOM,
and container restarts remained zero. PostgreSQL was `Ready` with SameZone HA
`Healthy`, and no external stop occurred.

R13 later failed at 2026-09-02T06:56:29Z after 47,540,000 committed rows and
zero rejects. Its durable resume token locates the failure inside vertex
mapping 4 (`Shipment`) after 41,540,000 rows in that mapping. The loader maximum
RSS was 907,872 KiB with zero swap. The source Java process reached
approximately 80.9 million KiB RSS and again emitted `OutOfMemoryError` after
progressively longer full-GC pauses, without a kernel/cgroup OOM kill or
container restart. Doubling the heap moved the failure from approximately
20.7 million to 41.54 million rows inside the same single mapping, proving that
the unbounded lifetime of one generated Bolt query—not the loader or target—is
retaining server-side state proportional to rows consumed.

The stopped r13 container and its logs were retained with a guest-side checksum.
All three VMs are deallocated and the Flexible Server stop is in progress while
the correction is validated. Generated discovery queries now end each ordered
stream after `LIMIT $pageRows`; the iterator closes it and resumes from the
last unique key, bounding each server-side auto-commit query to the configured
5,000 rows. The 48 GiB heap, 28 GiB page cache, source data and image, 8-byte
identity representation, loader limit, and target settings are unchanged. The
complete Go test suite passes. R13 is never resumed; r14 uses a fresh database,
job, and query fingerprint.

Fresh run `clean-r14-neo4j526` started at 2026-09-02T08:11:31Z on commit
`9391b23c7527ea35338fa731eb18a88136efaa65`. The replacement source proved a
48 GiB initial/maximum heap, 28 GiB page cache, online read-only access, and
zero swap/OOM events; the stopped r12 and r13 containers remain retained.
PostgreSQL 18.6 was `Ready` with SameZone HA `Healthy`, the loader had 41% disk
use and no competing P3 unit, and the validated source-before profile was
reused. R14 prepared a fresh target and is executing the corrected load; its
durable job row has not yet been created.

R14 created durable job `01e74add-b171-4cc7-96ad-44a5d1035ffc`. At
2026-09-02T08:49:23Z it had committed 41,420,000 rows with zero rejects, next
batch 2,072, and a current checkpoint. It has passed the 26.7-million-row
failure boundary shared by r10, r11, and r12 and is approaching the retained
r13 boundary at 47.54 million rows. Loader RSS was 830,448 KiB, below the
4 GiB gate. Source Java RSS was 80,756,788 KiB with approximately 50.5 GB host
memory available; source and loader swap, kernel OOM, Java OOM, and container
restarts remained zero. Loader and source data-disk use were 41% and 32%.

At 2026-09-02T09:54:13Z r14 had committed 131,940,000 rows with zero rejects,
next batch 6,598, and a 342-second checkpoint age. It has progressed to almost
three times r13's 47.54-million-row failure boundary with no Java, kernel, or
cgroup OOM, no container restart, and no swap. Loader RSS was 1,677,352 KiB,
below the 4 GiB gate. Source Java RSS was 80,794,316 KiB with approximately
50.4 GB available on the source host. Azure Flexible Server storage was 39.21%,
loader/source data-disk use was 41%/32%, and the posted P3 actual cost was
290.22 USD against the 800 USD ceiling. PostgreSQL remained `Ready` with
SameZone HA `Healthy`; no external governance stop occurred.

At 2026-09-02T11:50:54Z r14 had committed 299,860,000 of 560,000,000 rows
(53.5%) with zero rejects, next batch 14,994, and a current checkpoint. Loader
RSS was 2,732,008 KiB, still below the 4 GiB gate; source Java RSS was
80,917,596 KiB with approximately 50.3 GB host memory available. Swap, OOM
events, and container restarts remained zero. Azure installed OS updates on the
loader and Neo4j 5.26 VMs from 11:04:40Z to 11:08:00Z. Both guests retained
their 08:06 boot times, the loader unit retained its 08:11:31 activation time,
the source container retained its 08:07:24 start time and zero restart count,
and checkpointed progress continued; therefore the external patch action did
not interrupt the clean run. Flexible Server storage was 42.06%, below the 80%
gate. The cost endpoint was rate-limited; the latest posted value remains
290.22 USD against the 800 USD ceiling.

R14 committed all 560,000,000 rows with zero rejects and 28,000 completed
batches at 2026-09-02T13:48:53Z. The uninterrupted load completed in 5:37:01
with exit status zero and maximum loader RSS 2,733,644 KiB, below the 4 GiB
gate. The source remained online and read-only with no swap, OOM event, or
container restart. The same service immediately advanced to the exact job
report and count collection; at 13:53:18Z that report was still running and no
post-load gate had failed. Flexible Server storage was 46.40%, below the 80%
limit. Preserve the active service and allow the complete post-load and digest
sequence to continue.

By 2026-09-02T14:49:21Z the exact job report, built-in verification, exact
source-after profile, doctor, pre/post-`ANALYZE` optimization reviews, target
`ANALYZE`, and deterministic fixture digest had all completed successfully.
The full target canonical digest then started and was active at 14:51:20Z.
Source Java RSS remained stable at 81,028,300 KiB with approximately 50.2 GB
host memory available; swap, OOM events, and container restarts remained zero.
Flexible Server storage was 46.35%, within the 80% gate. The cost endpoint was
still rate-limited, so the latest posted actual remains 290.22 USD against the
800 USD ceiling. Do not interrupt the target digest.

At 2026-09-02T15:06:49Z the external Azure governance principal stopped the
Flexible Server while the first r14 target-digest attempt was active. It then
deallocated the loader and Neo4j 5.26 VMs at 15:47Z. The digest emitted no
target output; the committed job and all completed clean-run evidence remain
retained. This was not an authorized qualification fault. The posted P3 actual
cost was 353.61 USD, within the 800 USD ceiling, and the last storage reading
was 46.35%, within the 80% gate.

Only the loader and Flexible Server were restarted; the source VMs remain
deallocated. After the private-network startup window cleared, PostgreSQL was
`Ready` / HA `Healthy`, the same job again proved `committed` with 560,000,000
rows and zero rejects, no competing P3 unit existed, loader disk use was 41%,
and swap/OOM remained zero. Core clean-r14 evidence checksums were recorded and
the interrupted attempt had produced no partial target digest. Retained-job
digest retry `clean-r14-digest-r2-neo4j526` started at
2026-09-02T15:58:49Z against verifier commit
`9391b23c7527ea35338fa731eb18a88136efaa65`; it was active at 15:59:40Z.

Digest retry r2 completed at 2026-09-02T18:13:03Z with exit status zero after
2:14:14. It covered all 560,000,000 rows in 5,600 leaves; every leaf and the
final root matched the retained expected root
`0302c456d17c6e9ee64552d68e2bf6a775e63cd3b09120f5bc342d329bddd1ba`.
Maximum verifier RSS was 2,639,072 KiB, with zero swap or OOM event. Report and
doctor outcomes are `pass`; bounded integrity and optimizer reports contain no
failed or unknown checks beyond their documented incomplete coverage, which
the full canonical digest closes. After excluding timestamps and query-paging
execution metadata, the source before/after profiles are identical with
semantic SHA-256
`3cb344c0da43fe8891261a7eb8ff2f158570d64259bd3179b01b1c687af1d88f`.
The Neo4j 5.26 clean qualification is complete.

The Neo4j 4.4 source VM then started for the recovery qualification. At
2026-09-02T18:20:07Z it proved Neo4j Community 4.4.48, an online read-only
database, 24 GiB initial/maximum heap, 28 GiB page cache, 38 online indexes,
32% data-disk use, and zero swap/OOM/container restarts. Fresh recovery segment
`recovery-r1-segment1-neo4j44` started at 18:22:07Z on commit
`9391b23c7527ea35338fa731eb18a88136efaa65`, prepared isolated targets, reused
the accepted clean source-before profile, passed validation and target
preflight, and entered the initial load. Its durable job row had not yet been
created at 18:23:01Z.

### Recovery qualification

| Step | Fault/recovery evidence | State |
| ---: | --- | --- |
| 1 | Start a fresh target database and durable job | PENDING |
| 2 | Block PostgreSQL connectivity near 60% and retain the checkpoint | PENDING |
| 3 | Restore connectivity and resume the same durable job | PENDING |
| 4 | Reboot the loader VM near 75% | PENDING |
| 5 | Explicitly resume the same durable job and finish the load | PENDING |
| 6 | Run the complete built-in post-load checks | PENDING |
| 7 | Compute the complete target digest and match fixture and clean roots | PENDING |

The recovery run reuses the accepted clean source profiles as the source
immutability proof, but it must compute a new complete target digest.

## Live guardrails

| Gate | Latest observed state |
| --- | --- |
| Live window | Within the authorized 96 hours; extended from 72 hours on 2026-09-02 |
| Cost | Posted actual value: 353.61 USD; ceiling: 800 USD |
| Flexible Server storage | Azure last reported 48.85%; limit: 80% |
| PostgreSQL / HA | PostgreSQL 18.6 `Ready`; SameZone HA `Healthy`; private authentication passed |
| Loader memory | Pre-fault RSS 2.31 GiB; resumed process RSS 0.93 GiB; limit: 4 GiB |
| Swap / OOM | Current recovery loader/source swap and OOM zero; retained failed-run evidence unchanged |
| Active resources | Loader, Neo4j 4.4 VM, and Flexible Server running; Neo4j 5.26 VM deallocated |
| External actions | Governance stop of the first r14 digest retained; later loader OS update completed without interrupting successful retry r2 |

The complete acceptance and stop criteria are defined in
[`../../docs/acceptance.md`](../../docs/acceptance.md), and the authorized P3
sequence is defined in [`../../docs/p3-run-sheet.md`](../../docs/p3-run-sheet.md).
