# Independent Neo4j network recovery — preparation

Status: **target and AGE ready / resized runner ready / preparing network trial / not qualified**.
This is a separate B11 trial; the nine
base routes and CSV recovery r2 retain their existing acceptance evidence.

## Scope and boundaries

- Workflow `8a9ae99e-c621-4a94-afd1-a30ff210a201`, source
  `af-n526-source`, name `neo4j526-network-recovery-p1-r1`.
- Existing dedicated trial group, Japan East / zone 1, existing runner subnet.
- Current user ceiling USD 800, outer deadline
  `2026-09-20T07:14:35.311Z`; conservative reserve USD 600, not measured billing.
- The new runner's daily shutdown is 07:00 UTC. A direct schedule read on
  September 16 confirmed the existing source's shutdown is **16:00 UTC**, not
  07:00 UTC; the source schedule was not changed. Do not start a fault trial
  that cannot safely reach a retained checkpoint before the runner boundary.
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

## Inventory attempt 1 — authentication rejected

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
authorize resetting this source. The schedules are recorded in Scope above.

## Authorized original-source password recovery

The user explicitly approved resetting **this original AZ-N526 source** and
saving the new password before proceeding. Fresh checks confirmed the exact
subscription/resource/container, no RG locks, no intervening write/delete
governance events in the queried window, disk 6%, no swap/OOM and no active
assessment. The trial's USD 800 ceiling and outer deadline are unchanged.

A random 32-byte password was saved and read back through the macOS Keychain
API before cloud mutation. **Service/label: `agefreighter-az-n526-neo4j`;
account: `neo4j`.** This is separate from the OP-N526 item; that clone and its
credential were not modified. The new secret was supplied as an Azure protected
parameter through stdin, never as command-line arguments, chat output or a
local request/password file. The local helper refuses to replace an existing
Keychain item and must not be blindly rerun.

Managed command `af-az-n526-password-reset-20260916` ran 05:36:00–05:36:57 UTC
and finished Succeeded / exit 0. The reviewed recovery script SHA-256 is
`c219538e7c55eed23de0d2cf449eba7f37d4b312bcf0fbb8060f04d43852a8e6`.
It stopped the normal container, backed up the system database, used an
unpublished loopback-only recovery container, reset only the native account,
and restored normal authenticated service. Authenticated reads returned
**1,600,000 vertices and 4,000,000 edges**. This is a count check, not a new
canonical digest qualification.

Guest evidence remains at
`/var/lib/agefreighter/neo4j-password-recovery/20260916T053600Z`:

- System backup SHA-256:
  `27f09ec9d078999745c5505da1a9e96cdb4a893a88248f0733bf8a04f0362786`;
  independent checksum verification passed.
- Summary SHA-256:
  `84e260bc8e6d01c60a242c37d67b710d5490107a03e86c149177d7726d5d0465`.
- Independent post-check: normal container running, same image and port
  bindings, zero restarts/OOM, disk 6%, swap 0.

After success and independent evidence verification, the temporary managed
command containing the protected credential was deleted; a fresh list confirms
it is absent. The guest backup/logs and Keychain item are retained.

The installed GUI refreshed same-boot idle readiness at 05:36:38 UTC, retained
failed inventory `10645afd-82c6-464c-a181-f57348609cbd` in history, and prepared
a new inventory without replaying it. The next inventory is waiting for the
new Keychain credential at the protected VS Code password prompt. No target
deployment, migration or network fault has started.

## Fresh installed-GUI inventory PASS

After the user supplied the new AZ-N526 Keychain password, the GUI refreshed
Linux readiness and submitted new operation
`b9f584c1-fbdd-427e-bfa4-68aa19ce1de5` without replaying the retained failure.
The accepted boot and public source configuration are unchanged; guest
configuration SHA-256 is
`103e58a065b1455bf553815f55cf0eadda6e5cf33e74ffc5d1f7c2b4d782165d`.

The GUI completed its create-only export/import flow and displayed the
hash-verified inventory. Independent local readback confirms 663 bytes,
SHA-256 `ade134ea10f84f25573b88cb38cfcc86df10763334cb72f62686b16711688509`,
report generated at `2026-09-16T05:42:48.019759695Z`, outcome PASS, no errors
or incomplete checks, and exact transactional count-store totals of 1,600,000
vertices plus 4,000,000 edges. This remains source inventory, not migration
qualification.

Target review selects a fresh private PG18 / AGE server
`afpg-8a9ae99ec6214a94afd1`, D4ds_v5 / 128 GiB, and later same-runner D4s_v5
resize. Fresh VNet inspection found `10.246.20.0/24` non-overlapping; no subnet
has been created. The outer deadline initially exceeded the product's maximum
96-hour per-plan horizon. The review therefore uses the shorter per-run deadline
`2026-09-17T07:00:00Z`, with the USD 800 ceiling and USD 600 conservative reserve
unchanged. This does not alter the outer authorization or auto-shutdown settings.
A subsequent freshness check correctly refused stale guest readiness; refresh
and complete the native review before saving or deploying. No target credential,
target deployment, resize or migration has been created at this checkpoint.

Fresh readiness at `2026-09-16T05:49:48.095Z` resolved that gate. The next
preflight rejected DDSv5 quota. Independent PostgreSQL quota reads confirmed
regional 62/196 cores, DDSv5 62/64 (only 2 free), and EDSv5 0/256. This is a
real family-quota limit, not the separate Compute quota. Existing accepted
servers were not deleted or resized to free quota. Review is being repeated
with E8ds_v5 / 128 GiB in the same region/zone, retaining the D4s_v5 runner,
shorter deadline, ceiling and reserve. The alternative passed native preflight
and its final confirmation displays **USD 1.448/hour combined compute** plus
the USD 600 reserve. The new target/credential creation was presented for
action-time approval; the native dialog remains pending. No target credential,
saved target plan, deployment or resize exists yet. If approval is delayed,
recheck readiness and price through the product before submission; do not
modify workflow state to bypass freshness.

## Authorized target submission

The user's continuation approved the exact alternative target/credential plan.
The first save correctly stopped before Azure submission because guest freshness
expired during folder selection. Its secret-reference-only files are retained;
the deployment was not replayed. Linux readiness was refreshed at
`2026-09-16T06:06:58.593Z`, and the same native review was completed with a new
create-only output directory. Plan SHA-256:
`6675b2b57e3cd95e492e9519843379fd54aafa2f52cb8d92bcd015723193c552`.

The installed GUI submitted deployment `afpg-8a9ae99ec6214a94afd1` at
approximately `2026-09-16T06:09:33Z`. Independent ARM reads show Running,
with the dedicated subnet/private DNS/link succeeded and server provisioning
active. This is E8ds_v5 / 128 GiB / PG18, private-only, zone 1; the approved
USD 1.448/hour combined compute, USD 600 reserve, USD 800 ceiling and September
17 07:00 UTC per-run deadline are unchanged. No load, fault or runner resize
has started. Creation status is not AGE readiness or qualification.

ARM completed Succeeded at `2026-09-16T06:16:24.376887Z` after 6m51s. The
installed GUI reconciled the exact deployment to provisioned without replay.
Independent server inspection confirmed version 18, E8ds_v5, 128 GiB, zone 1
and public access Disabled. The reviewed AGE preload restart was submitted
at `2026-09-16T06:17:02.998Z`; migration remains unstarted. Saved LoadJob and
target plan permissions are 0600; structural inspection found no literal secret
fields. Source ARM NIC readback also confirms the observed `10.246.5.4` has
no public IP. No RG locks or delete/stop/deallocate activity was found in the
queried deployment window.

The GUI reconciled AGE restart to finished; independent readback confirms
`shared_preload_libraries=pg_stat_statements,age` with no pending restart. With
fresh idle health at 06:17:28 UTC, the installed GUI performed each explicit
same-VM resize step: deallocate, reconcile, change to D4s_v5, reconcile, start.
The retained disk/NIC/identity/placement seal is
`bbb092b27493316992059c86a7fc00651f6e8bf894447a8aed38d61cc1ac99ac`.
At 06:21 UTC, independent ARM inspection confirmed D4s_v5 / Succeeded /
Running. The Mac then reported locked, so no further GUI interaction was
attempted. The local workflow correctly remains `resize.phase=starting`
until a read-only GUI reconciliation; it was not manually advanced.

Next after manual unlock: reconcile resize, acquire fresh post-boot Linux
readiness, review migration and request the source password privately. There
is **no migration job, network fault, or automatic resume** yet. The new
runner's daily 07:00 UTC shutdown remains unchanged. If the interaction is
delayed beyond that boundary, recheck actual state and scope before proceeding;
this document is not a background monitor or a permission to bypass health gates.

## Unlock and scheduled-stop reconciliation

After the user unlocked the Mac, the 07:00 UTC daily shutdown had taken effect.
Activity Log confirms deallocation completed at 07:01:19 UTC by the service
principal independently identified as **Azure Lab Services**; the exact enabled
schedule is 07:00 UTC. No unexpected stop/delete/write or RG lock was found in
the checked windows. Under the unchanged trial authorization, only the same
runner was restarted. The daily schedule was not disabled or changed.

The installed GUI reconciled resize to finished and verified new boot
`8c9dcd46-c99f-4dd5-8966-4cf737d2e485` at 07:09:32 UTC: the same pinned
CLI/commit/archive, idle guest, 4.094% disk, zero swap/OOM. Source VM remains
running and the private target remains Ready. There is still no migration job.

`await-network-load.py` is a one-workflow, 15-minute guest test helper to avoid
missing a fast P1 boundary during private password entry. It cannot start or
resume a migration; it refuses any pre-existing load, binds only the next fresh
operation to this exact workflow/source/graph/boot and private target, seals the
selected job/config hash, then calls the reviewed bounded observer. It is not
a recurring monitor or extension product feature. Three additional local tests
pass for exact binding and refusal of changed/multiple jobs. Actual deployment,
arming, fault and recovery evidence must be recorded separately.

## Observer preparation

The retained guest observer now supports explicit **read-only** Neo4j
observation. It binds the hashed configuration's actual source type and exact
`migrate-source`/`resume-migration` command arguments, retaining existing boot,
job, fingerprint, checkpoint, memory, disk, swap and OOM checks. Its process and
reboot fault switches remain CSV-only. An explicit `--network-source-ip`
option now prepares a narrowly scoped Neo4j connection-rejection trial:

- Only the exact operation cgroup, the resolved private fixture address
  `10.246.5.4`, and TLS Neo4j port 7687; no source/NSG/firewall flush changes.
- First fresh load only, 1.4M to less than 2.5M committed rows, with matching
  job/generation/fingerprint/config/boot, healthy resource bounds and checkpoint.
- Create-only pre-fault evidence prevents retry of an ambiguous attempt.
- An independent systemd timer is armed before insertion to remove the exact
  rule at 45 seconds. The normal path removes it after 5 seconds in `finally`.
  Admission refuses an expired timer margin; both applied and absent-rule
  evidence are sealed. No automatic migration resume is performed.

All **18 local observer tests PASS**, including refusal of wrong destination,
cgroup, transport, identity and replay, and restoration after an evidence-write
failure and expired/unavailable restoration timers. A read-only guest check at
06:13:39 UTC confirmed cgroup2, iptables cgroup path support, source DNS
`10.246.5.4`, disk 4% and zero swap. Actual kernel rule installation, fault,
explicit same-job resume and canonical verification remain separate live gates.
No network fault has yet occurred.

## Remaining admission and acceptance

Runner approval, readiness and selected-source start are now recorded above.
Next: reconcile the submitted target and complete AGE readiness and same-VM
resize. Fresh source inventory is accepted above; the later load still requires
source credentials through the protected GUI. Recheck
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
