# P3 production-simulation progress

This is the redacted live progress report for the P3 qualification in
`rg-afps-p3-20260831`. It is updated at material phase transitions; retained
guest evidence and the final reviewed result remain authoritative.

- Updated: 2026-09-01T09:20:52Z
- Overall state: **CORRECTIVE VERIFICATION RETRY**
- Current position: **Neo4j 4.4 clean — target digest verifier correction**
- Next: deploy the corrected verifier and rerun only the full target canonical
  range digest against the retained committed clean database
- Target: PostgreSQL 18 / Apache AGE 1.7 on Azure Database for PostgreSQL
  Flexible Server

## Qualification overview

| Source | Qualification run | State | Current or next step |
| --- | --- | --- | --- |
| Neo4j 4.4.48 | Clean | **CORRECTING** | Step 9 of 10: target digest verifier retry |
| Neo4j 4.4.48 | Recovery | PENDING | Start only after both clean-source qualifications |
| Neo4j 5.26.30 | Clean | PENDING | Start after the complete 4.4 clean evidence |
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
| 9 | Full target canonical range digest | **CORRECTING** | The first verifier treated the operational Neo4j internal ID as the visible `external_id`; the committed target is retained and only the verifier is being corrected and retried |
| 10 | Range/root comparison and run summary | PENDING | Must match every range and the final Merkle root |

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
| 1 | Power on the retained source VM and prove version, zone, read-only state, and exclusivity | PENDING |
| 2 | Target preflight, plan, and exact source profile before load | PENDING |
| 3 | Uninterrupted 560-million-record migration | PENDING |
| 4 | Job report and exact count collection | PENDING |
| 5 | Built-in counts, catalog, generation, and bounded integrity checks | PENDING |
| 6 | Exact source profile after load | PENDING |
| 7 | Doctor, optimization review, target `ANALYZE`, and post-`ANALYZE` review | PENDING |
| 8 | Deterministic fixture manifest and expected range digest | PENDING |
| 9 | Full target canonical range digest | PENDING |
| 10 | Range/root comparison and run summary | PENDING |

The Neo4j 5.26 VM is currently deallocated and no 5.26 migration unit exists.

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
| Live window | Within the authorized 72 hours |
| Cost | Last successfully posted value: 159.32 USD; ceiling: 400 USD |
| Flexible Server storage | 34.20%; limit: 80% |
| PostgreSQL / HA | Ready / Healthy |
| Loader memory | 2.61 GiB peak; limit: 4 GiB |
| Swap / OOM | None |
| External actions | Two Azure patch actions retained as governance deviations; neither restarted the qualification processes |

The complete acceptance and stop criteria are defined in
[`../../docs/acceptance.md`](../../docs/acceptance.md), and the authorized P3
sequence is defined in [`../../docs/p3-run-sheet.md`](../../docs/p3-run-sheet.md).
