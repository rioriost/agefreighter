# Independent Neo4j network recovery — preparation

Status: **runner ready / source inventory awaiting credentials / not qualified**.
This is a separate B11 trial; the nine
base routes and CSV recovery r2 retain their existing acceptance evidence.

## Scope and boundaries

- Workflow `8a9ae99e-c621-4a94-afd1-a30ff210a201`, source
  `af-n526-source`, name `neo4j526-network-recovery-p1-r1`.
- Existing dedicated trial group, Japan East / zone 1, existing runner subnet.
- Current user ceiling USD 800, outer deadline
  `2026-09-20T07:14:35.311Z`; conservative reserve USD 600, not measured billing.
- Existing daily 07:00 UTC VM shutdown remains unchanged. Do not start a fault
  trial that cannot safely reach a retained checkpoint before that boundary.
- No accepted graph may be reused, faulted or overwritten. Preserve all
  failed-run evidence, original identity/configuration and explicit GUI resume.

## September 16 storage and artifact preparation

The user requested continuation after the exact storage/role confirmation.
At approximately 05:13 UTC the unchanged native confirmation was approved.
Deployment `af8a9ae99ec6214a94afd1a3-transfer` completed successfully.
Independent ARM reads verified ownership/workflow tags and one exact-account
Storage Blob Data Contributor assignment for the signed-in user.

Azure initially reported public network access Disabled. Under the previously
authorized **trial-storage-only** exception, `SecurityControl=Ignore` and the
outer expiry were merged only onto the new storage account; the reviewed
authenticated HTTPS endpoint was restored to Enabled. Anonymous access and
shared keys remain false, HTTPS-only and TLS 1.2 remain enforced. An unauthenticated
account-list probe returned HTTP 409. No source firewall, VNet, RG-wide security
exception or accepted target was modified. No specific external actor was
attributed: the initial Activity Log query had not returned events.

The installed GUI reconciled storage to `ready / Enabled`. It then selected,
hashed and uploaded the same locally reviewed Linux archive used by accepted
CSV recovery r2, without executing it:

| Field | Value |
|---|---|
| CLI | `2.4.0-dev.7538981cf0fc` |
| Commit | `7538981cf0fc6c1bed3a50e6476861e84647a003` |
| Archive bytes | 37,117,370 |
| SHA-256 | `6f10538cc70c2125cc669a4d043352efdfce510131e1bec440af3674a4a48d21` |
| Destination | This workflow's container in `af8a9ae99ec6214a94afd1a3` |

Local workflow metadata confirms `developmentUpload.phase=ready` and the exact
archive seal. This is not runner installation, source access or migration.
GUI runner preflight is the next gate. Fresh regional quota reads were 50/101
cores and DSv5 48/100; these are subscription totals, not ownership evidence
for this trial and not a guarantee of target capacity.

The installed GUI then passed the runner preflight and generated a preview for
`af-8a9ae99ec6214a94afd1`: Standard_B2s_v2, Japan East / zone 1, compute
USD 0.109/hour plus disk/network/storage. This preview expires at
`2026-09-16T05:32:52.109Z` (not the trial's outer deadline). Its exact native
final creation dialog is pending action-time approval of the pinned unpublished
installation and the new VM identity's Blob Reader grant on this workflow's
container only. No VM deployment has been submitted. Recheck/renew the preview
through product controls if it expires; never bypass expiry or edit saved state.
All six retained VMs and fifteen Flexible Servers were still stopped at that
preflight inventory. The following checkpoint supersedes the pending-dialog
and source-stopped state above.

## September 16 runner readiness and source start

At the next GUI observation, the workflow already showed deployment submitted
at 05:20:37 UTC. No duplicate deployment was submitted. Independent ARM reads
confirmed deployment/VM success at 05:21:19 UTC: the exact new B2s_v2 runner,
zone 1, no public IP. Its managed identity has one observed Blob Data Reader
assignment, scoped only to this workflow's container.

The installed GUI reconciled provisioning and executed Linux readiness at
05:23:32 UTC. The retained report confirms the reviewed archive SHA/commit,
Neo4j inventory/migration and explicit-resume capabilities, idle guest,
3.48% storage use, zero swap and zero OOM events. ARM success alone was not
used as readiness evidence.

A daily 07:00 UTC safety shutdown was enabled for the new VM. Only
`af-n526-source` was started among existing fixtures. Its bounded read-only
guest check at 05:25:11 UTC found the retained Neo4j container running,
6% root-disk use, no swap and no kernel OOM messages since this start window.
The source firewall, credentials and graph were not changed. The USD 800
ceiling, USD 600 conservative reserve and outer deadline remain unchanged;
the reserve is not a measured bill. Existing daily shutdowns remain effective.

The source form still binds the exact hostname, TLS CA, `source_key` identities
and new workflow. The complete-inventory read was approved in the installed
GUI, bounded to 30 minutes / 4 GiB / no swap. Execution is waiting at the
protected password input; no source inventory, target deployment, migration
or network fault has started. Credentials must be supplied directly in VS Code,
not in a report or chat. Previously accepted targets remain untouched.

## Observer preparation

### Inventory attempt 1 — authentication rejected

After the user confirmed private password entry, the installed GUI refreshed
readiness and submitted operation `10645afd-82c6-464c-a181-f57348609cbd`.
Its configuration SHA-256 is
`4fc54297f4d7f845bc85abb5db554ada335252ddbb515ad5f8542ae8993f382e`;
guest configuration SHA-256 is
`e6bb0389e4c31e6fd91021da701c7ddf3ad7153555fc5bad3deede3bd7a3851a`.
It failed with exit 1 between `2026-09-16T05:26:40.136772960Z` and
`2026-09-16T05:26:40.461270949Z`. A guest-side classification of retained stderr
identified `Unauthorized`, not DNS, TLS, connection-refused or timeout errors.
No raw stderr or credential was exported. The exact operation remains retained;
no automatic retry, target creation, data load or fault was performed.

The September 12 AZ-N526 execution sheet explicitly records that the original
source credential was temporary and not persisted for reuse. The September 15
Keychain item `agefreighter-op-n526-neo4j` belongs only to the separately reset
OP-N526 clone. It must not be represented as the original source credential.
The next attempt requires the correct original credential, or explicit approval
for original-source credential recovery. Clone-only reset approval does not
authorize resetting this source. Running VMs retain the 07:00 UTC shutdown.

The retained guest observer now supports explicit **read-only** Neo4j
observation. It binds the hashed configuration's actual source type and exact
`migrate-source`/`resume-migration` command arguments, retaining existing boot,
job, fingerprint, checkpoint, memory, disk, swap and OOM checks. Its process and
reboot fault switches remain CSV-only; no network fault is implemented or
implicitly authorized by this change. All **10 local observer tests PASS**.
These tests do not prove a live network fault or resumed migration.

## Remaining admission and acceptance

Runner approval, readiness and selected-source start are now recorded above.
Next: obtain the source password through the protected GUI and reconcile the
complete source inventory. Target/resize review remains required; recheck
cloud/guest/time/cost gates before subsequent mutations.

Before fault injection: review the exact destination and job scope, an
independent bounded automatic restoration path, and retained before/after
evidence. Refuse stale checkpoints, unrelated active jobs, storage >=80%,
loader memory >4 GiB, swap/OOM, expired cost/time gates or unexplained governance
changes. Do not substitute a loader SIGTERM for network-loss evidence.

After restoration: use the native same-job inspection and explicit resume;
prove unchanged database/graph/job/generation/fingerprint and source data.
Require complete 5.6-million-record counts, zero rejects, all 64 canonical
ranges and the frozen P1 root before declaring success. Stop owned compute
after final evidence capture. Network-source recovery remains open until then.
