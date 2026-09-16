# Independent Neo4j network recovery — preparation

Status: **not started / not qualified**. This is a separate B11 trial; the nine
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
All six retained VMs and fifteen Flexible Servers were still stopped at the
fresh preflight inventory. The source fixture has not been started.

## Observer preparation

The retained guest observer now supports explicit **read-only** Neo4j
observation. It binds the hashed configuration's actual source type and exact
`migrate-source`/`resume-migration` command arguments, retaining existing boot,
job, fingerprint, checkpoint, memory, disk, swap and OOM checks. Its process and
reboot fault switches remain CSV-only; no network fault is implemented or
implicitly authorized by this change. All **10 local observer tests PASS**.
These tests do not prove a live network fault or resumed migration.

## Remaining admission and acceptance

Before starting: approve the exact new runner identity/container access and
pinned installation, verify guest readiness, start only the selected fixture
source after cloud/guest gates pass, and obtain its password through the
protected GUI. Complete source inventory and target/resize review are required.

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
