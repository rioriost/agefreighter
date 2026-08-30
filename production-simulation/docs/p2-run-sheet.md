# P2 reviewed run sheet

## Authorization and resource envelope

The user authorized P2 execution on 2026-08-30. This authorization covers the
P2 deployment, clean/tuning migrations, four separately injected recovery
faults, replace qualification, verification, monitoring, and stopping the
resources. It does not authorize P3 or resource deletion.

P2 uses a new `rg-afps-p2-20260830` resource group in Japan East zone 1. The
maximum live window is 24 hours. Stop without deleting evidence if the live
window or any criterion in `acceptance.md` is exceeded.

| Role | P2 reviewed value | P1 basis |
| --- | --- | --- |
| Neo4j source, each | `Standard_E8bds_v5`, 8 vCPU / 64 GiB; 24 GiB heap / 28 GiB page cache | P1 source CPU <10%; P2 store is 22-23 GB; P2 `neo4j-admin` recommendation |
| Source data disk, each | 512 GiB Premium SSD v2, 10,000 IOPS, 750 MB/s | P1 used <1% of 4 TiB and <=12% of 40,000 IOPS |
| Loader | `Standard_D8ds_v5`, 8 vCPU / 32 GiB | P1 CPU 11.65% and RSS about 105 MiB |
| Flexible Server | `Standard_E8ds_v5`, Memory Optimized, SameZone HA | P1 CPU 3.73% and memory 31.88% on E32 |
| Target storage | 1,024 GiB, autogrow disabled | P1 coefficient ~42.4 GB per P2 graph; preserves multiple runs |

All resources remain private, in the same region and availability zone. The
resource-hour ceiling is two source VMs, one loader VM, one HA Flexible Server,
two 512 GiB source disks, and 1 TiB target storage for at most 24 hours. Azure
Cost Management may post after the run; the exact billed cost is appended when
available.

## Tuning sequence

Only Neo4j 4.4.48 is used to select the setting; the frozen winner is then run
cleanly against both source versions. Each tuning candidate gets a fresh
database and changes only one setting from baseline:

| Candidate | `fetchRows` | `batchRows` | `batchBytes` |
| --- | ---: | ---: | ---: |
| baseline | 5,000 | 10,000 | 64 MiB |
| fetch | 10,000 | 10,000 | 64 MiB |
| rows | 5,000 | 20,000 | 64 MiB |
| bytes | 5,000 | 10,000 | 128 MiB |

The winner must satisfy every correctness and memory gate; elapsed time is the
tie-breaker. A configuration change is never applied to an existing job, so
resume fingerprints remain stable.

The baseline receives the full source-before/source-after profile and canonical
digest. The three one-variable tuning candidates still load all records and run
report, exact counts, bounded integrity, doctor, and optimizer checks, but do
not repeat the expensive source profiles or canonical digest. This keeps the
24-hour envelope available for the required clean, recovery, and replace runs.
The frozen winner must pass the full digest on both source versions and every
recovery result must match that clean root.

### Frozen result

The four Neo4j 4.4.48 candidates all committed 56,000,000 records with zero
failed batches and zero swap. The frozen setting is `fetchRows: 5000`,
`batchRows: 20000`, and `batchBytes: 64MiB`. It completed in 47:51.61, which is
187.82 seconds (6.1%) faster than the 50:59.43 baseline. Its maximum RSS was
179,480 KiB, its committed-batch p99/median latency ratio was 1.338, and the
last/first throughput ratios measured separately within vertices and edges
were 92.9% and 98.6%. The unsegmented ratio is not used because it compares
cheap vertex batches at the beginning with more expensive endpoint-resolving
edge batches at the end.

The first baseline attempt used 8 GiB heap / 40 GiB page cache and stopped at
5,460,000 committed records when Neo4j 4.4 reported
`Neo.TransientError.General.OutOfMemoryError`. agefreighter RSS was 102,376 KiB
and the loader did not swap. The failed database and raw result remain retained
under run ID `tune-baseline`; the rerun uses a new database and run ID. Source
memory was corrected to the measured `neo4j-admin` recommendation before any
further tuning comparison.

## Recovery schedule

Every recovery run uses a fresh database and the frozen settings. Faults are
separate, occur only after a committed checkpoint, and are divided across both
supported sources:

1. Neo4j 4.4.48: terminate agefreighter at approximately 25%, then resume.
2. Neo4j 4.4.48: stop/restart the Neo4j container at approximately 40%, then resume.
3. Neo4j 5.26.30: block/restore PostgreSQL connectivity at approximately 60%, then resume.
4. Neo4j 5.26.30: reboot the loader VM at approximately 75%, then resume the
   still-running durable job.

Each recovery target must match the clean-run canonical root. Checkpoints must
advance monotonically, and recovery must begin within 15 minutes after the
fault clears.

Start each fault run as a detached system service so the operator can inject
the fault through a separate control session. After the faulted process exits,
query the durable job ID from `agefreighter_meta.load_job` and start a new
evidence run with `P2_RESUME_JOB_ID`. The resume run must use the same source,
target database, and frozen settings, and must not prepare or recreate the
target database. Preserve both the interrupted and resumed result directories.

## Replace qualification

After the clean create run, perform a full `replace` against one reviewed P2
target. Verify that the old graph stays active before promotion, the replacement
commits atomically, the retained backup is reported, and the replacement root
matches the clean root. Do not run `cleanup`; backup deletion is outside this
authorization.

Use `P2_TARGET_MODE=replace` for the replacement evidence run. It must reuse
the reviewed clean-run database with target preparation disabled.
