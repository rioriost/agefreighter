# P3 production-simulation progress

This is the redacted live progress report for the P3 qualification in
`rg-afps-p3-20260831`. It is updated at material phase transitions; retained
guest evidence and the final reviewed result remain authoritative.

- Updated: 2026-09-02T03:40:43Z
- Overall state: **RUNNING**
- Current position: **Neo4j 5.26 clean — fresh r12 uninterrupted load is running**
- Next: obtain the new durable job ID, then monitor checkpoints past the two
  former 26.7-million-row source-JVM failure boundaries
- Target: PostgreSQL 18 / Apache AGE 1.7 on Azure Database for PostgreSQL
  Flexible Server

## Qualification overview

| Source | Qualification run | State | Current or next step |
| --- | --- | --- | --- |
| Neo4j 4.4.48 | Clean | **DONE** | All 560,000,000 rows and 5,600 digest ranges match |
| Neo4j 4.4.48 | Recovery | PENDING | Start only after both clean-source qualifications |
| Neo4j 5.26.30 | Clean | **RUNNING** | Fresh r12 load active on the 128 GiB source VM; durable job creation is pending |
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
| 1 | Start a fresh target database and durable job | PENDING |
| 2 | Send loader `SIGTERM` near 25% and retain the checkpoint | PENDING |
| 3 | Resume the same database, graph, job, and fingerprint | PENDING |
| 4 | Restart the Neo4j 4.4 container near 40% | PENDING |
| 5 | Resume the same durable job again and finish the load | PENDING |
| 6 | Run the complete built-in post-load checks | PENDING |
| 7 | Compute the complete target digest and match fixture and clean roots | PENDING |

The recovery run reuses the accepted clean source profiles as the source
immutability proof, but it must compute a new complete target digest.

## Neo4j 5.26.30

### Clean qualification

| Step | Evidence gate | State |
| ---: | --- | --- |
| 1 | Power on the retained source VM and prove version, zone, read-only state, and exclusivity | DONE |
| 2 | Target preflight, plan, and exact source profile before load | DONE; retained for fresh retry |
| 3 | Uninterrupted 560-million-record migration | **RUNNING**; fresh r12 load process active |
| 4 | Job report and exact count collection | PENDING |
| 5 | Built-in counts, catalog, generation, and bounded integrity checks | PENDING |
| 6 | Exact source profile after load | PENDING |
| 7 | Doctor, optimization review, target `ANALYZE`, and post-`ANALYZE` review | PENDING |
| 8 | Deterministic fixture manifest and expected range digest | PENDING |
| 9 | Full target canonical range digest | PENDING |
| 10 | Range/root comparison and run summary | PENDING |

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
The uninterrupted load process is active; at this phase transition its durable
job row has not yet been created. The prior r10 and r11 jobs remain retained
and are not resumed.

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
| Cost | Posted actual value: 253.00 USD after Azure revision; ceiling: 800 USD |
| Flexible Server storage | 34.92%; limit: 80% |
| PostgreSQL / HA | PostgreSQL 18.6 `Ready`; SameZone HA `Healthy`; private authentication passed |
| Loader memory | 2.61 GiB peak; limit: 4 GiB |
| Swap / OOM | None |
| Active resources | Loader and Neo4j 5.26 VM running; Neo4j 4.4 VM deallocated |
| External actions | Six patch windows, one PostgreSQL stop, and one loader/source VM deallocation retained; no new external stop observed during r12 startup |

The complete acceptance and stop criteria are defined in
[`../../docs/acceptance.md`](../../docs/acceptance.md), and the authorized P3
sequence is defined in [`../../docs/p3-run-sheet.md`](../../docs/p3-run-sheet.md).
