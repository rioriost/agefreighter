# Independent Neo4j network recovery — preparation

Status: **r1 retained without a network fault; fresh r2 storage/artifact ready; network recovery not qualified**.
This is a separate B11 trial; the nine
base routes and CSV recovery r2 retain their existing acceptance evidence.

## R2 checkpoint — September 16, 08:54 UTC

The installed GUI created a fresh draft `b2c7214e-83f5-4613-b378-98d36e0cd97d`,
named `neo4j526-network-recovery-p1-r2`, namespace `n526_recovery_r2`, graph
`neo4j526_network_recovery_p1_r2`. The source remains the original
`af-n526-source`; its reviewed CA and stable `source_key` mappings are unchanged.
No local workflow metadata was edited and no completed graph was reused.

Following the user's exact storage/role approval, the installed GUI submitted
and reconciled `afb2c7214e83f54613b37898`. Independent ARM reads confirmed
Succeeded, matching workflow ownership, and one User Blob Data Contributor
assignment scoped to this account. Initially its public network access was
Disabled; the existing trial-storage-only exception and outer expiry were
merged only onto this account before restoring authenticated HTTPS access.
Anonymous access/shared keys remain false, HTTPS-only and TLS 1.2 enforced.
The GUI subsequently displayed ready / Enabled. The exact previously reviewed
37,117,370-byte loader archive (commit `7538981cf0fc6c1bed3a50e6476861e84647a003`,
SHA-256 `6f10538cc70c2125cc669a4d043352efdfce510131e1bec440af3674a4a48d21`)
was uploaded through the installed GUI; persisted development upload is ready.
Uploading an archive is not installation, source assessment or migration.

No RG locks were present. The activity window showed the expected storage
deployment/role operations and policy modify actions. Separate failed TLS
configuration writes concerned older PostgreSQL servers, not this new draft;
no actor attribution or changes to those retained servers were made. R1's
Flexible Server was independently confirmed Stopped at 08:38 UTC; its runner
and source had already been confirmed deallocated. All R1 evidence is retained.

The local watcher is now bound only to R2 and rejects R1's workflow/graph.
Observer tests passed 20/20 and binding tests 4/4. It has not been uploaded or
armed: confirm the planned target hostname against the final target review,
then recheck all cloud/guest/time/cost gates. No new VM/target/load has yet been
submitted. The same USD 800 ceiling, USD 600 conservative reserve (not measured
billing), and September 20 07:14:35.311 UTC outer deadline apply.

Runner preflight subsequently passed. The GUI generated a fresh review at
08:52:36.651 UTC for `af-b2c7214e83f54613b378`, B2s_v2 in Japan East / zone 1,
USD 0.109/hour compute plus additional charges. The final native creation
dialog is waiting for action-time approval of the pinned test installation and
this new identity's exact-container Blob Reader grant. No VM was submitted.
If the preview expires while waiting, regenerate it through the product and
recheck that identity, artifact and scope are unchanged before proceeding.

Next: obtain the action-time installation/scoped-identity approval,
provision and assess, review a fresh target and same-VM resize, then
exercise the corrected watcher and explicit same-job GUI recovery. Require
exact counts plus all 64 canonical ranges before claiming network recovery.

### R2 runner ready — September 16, 09:00 UTC

On continuation, the native GUI already showed the approved deployment
submitted at 08:54:48.810 UTC. No duplicate deployment was submitted. GUI
reconciliation at 08:55:22.719 confirmed provisioning. Independent ARM reads
confirmed B2s_v2 / zone 1, matching workflow ownership, private IP `10.246.1.19`
on the existing runner subnet and no public IP. Its new managed identity has
one observed Blob Data Reader assignment on only the R2 workflow container.

The first readiness operation `3919ad79-b7c1-466b-b579-de10d773cdaa` exited 127
at 08:55:39: `/usr/local/bin/agefreighter-tools` was not yet installed. That
failure is retained, not treated as readiness. A bounded guest inspection at
08:57:03 found cloud-init done, the tool present, root disk 4%, zero swap and
zero kernel OOM matches. A separate explicit GUI readiness read
`8097ffad-db4a-4d44-973a-b1072a03e7e3`, submitted at 08:57:36.391, then passed:
the pinned archive/commit match, boot `d393b504-1347-4656-b40d-fa881747d535`,
Neo4j inventory/migration and explicit-resume capabilities, idle health,
3.5087% storage, zero swap and OOM. This is readiness, not migration acceptance.

The new runner has a verified Enabled daily 07:00 UTC shutdown. Only the
existing original `af-n526-source` was started among fixtures, with its unchanged
16:00 UTC shutdown, private `10.246.5.4`, no public IP and unchanged source
network/credentials. The pre-start inventory confirmed seven existing VMs
deallocated and all sixteen Flexible Servers Stopped. No target was created.

The installed GUI reviewed the same R2 source/CA/identity mappings and approved
the read-only count inventory (30 minutes / 4 GiB / no swap). It is now waiting
at the protected source password input; no inventory or migration has been
submitted. Password entry is direct in VS Code, never chat or a result file.
The watcher remains unarmed and all R1 evidence remains preserved.

### R2 source inventory PASS — September 16

After protected password entry, the installed GUI submitted inventory
`e2a61e33-7dff-4bf1-a501-40a505a3e11f` on the same accepted boot. Its guest
configuration SHA-256 is
`57d4f2a3682c7b6e99aac32a43be9947673baf475617953b545001a7d4218a7d`.
The create-only export/import completed in the GUI. Independent local hashing
confirmed the 663-byte report SHA-256
`da7881faba550835775b2da89b3776cb3ba2fa750f047549b70e46edd164a2cb`,
generated at 09:01:01.779990223 UTC: PASS, no errors/incomplete checks,
1,600,000 vertices and 4,000,000 edges from the transactional count store.
This qualifies inventory only, not the recovery migration.

The original source's bounded guest inspection at 08:59:50 UTC confirmed its
retained container running with the same pinned image, disk 6%, zero swap
and no OOM matches. A fresh VNet read found `10.246.21.0/24` unused. PostgreSQL
quota is 70/196 regional cores, DDSv5 62/64 and EDSv5 8/256, so the proposed
fresh target uses E8ds_v5 / 128 GiB and D4s_v5 on the existing runner.
The first target preflight correctly refused stale readiness; no target plan,
credential, subnet, resize or deployment was created. Repeat an explicit
readiness check and the native review without bypassing freshness. The proposed
per-run deadline remains September 17 07:00 UTC (earlier than outer permission),
with USD 800 ceiling and USD 600 conservative reserve.

Fresh same-boot GUI readiness at 09:05:34.601 UTC resolved the freshness gate.
The repeated target preflight passed, and the native final review displays
5,600,000 mapped rows, matching inventory SHA, E8ds_v5 / 128 GiB, D4s_v5 runner,
`10.246.21.0/24`, no public access/peering, USD 1.448/hour combined compute,
the unchanged USD 600 reserve and USD 800 ceiling, and September 17 07:00 UTC
deadline. It is waiting for action-time approval to create the new target and
its SecretStorage-held connection credential. No target plan was saved and no
target deployment or resize was submitted. If approval is delayed, repeat
readiness/preflight through the GUI; never edit retained state to bypass gates.

At the user's explicit target approval, the final native review was accepted.
The macOS folder dialog initially did not accept Return; raising that dialog
and using keypad Enter selected the intended R2-only local folder. No hidden
workflow edits or alternative deployment path were used. The GUI saved both
the secret-reference-only LoadJob and target plan with mode 0600 under
`production-simulation/work/network-recovery-r2-20260916/` (ignored local
evidence). The product's repeated preflight passed and submitted deployment
`afpg-b2c7214e83f54613b378` at 09:10:28 UTC, correlation
`3bb2a480-5024-41c6-bf6f-e46154e56ad1`. ARM currently reports Running.
Do not replay the submission. AGE readiness, same-VM resize, new load and
network recovery remain subsequent gates. The local watcher regressions were
re-run: 4 binding plus 20 observer tests pass; this is not live fault evidence.

The reviewed observer and R2 waiter were placed create-only in the new runner's
`qualification-network-tools` directory, after asserting no active operation
and no existing migration. Guest byte hashes match local:

- Observer: `eff22e0f7770c40343addba53328b2dbed548065db7a6182f071aff2beac2035`.
- R2 waiter: `128a09d1e5f0672dd3c20143f4b1ff240c3bdb149e1f342d2eba77eaf66d3735`.

Actual guest timestamp normalization passed all fractional precisions 1–9.
The watcher was not armed, and no network rule or fault timer was created.
ARM subsequently confirmed the new PG18 / E8ds_v5 server Ready with public
access Disabled, while the deployment was still applying AGE configuration.
Reconcile the entire deployment before treating the target as provisioned.

### R2 target and same-VM resize complete

Installed-GUI target reconciliation reached provisioned. The separate approved
AGE restart was submitted at 09:17:38.701 UTC and reconciled finished. A direct
parameter read confirms `pg_stat_statements,age` with no pending restart.
The source and accepted targets were untouched. The observed Azure policy
deployment applied this new server's advanced-threat-protection setting;
it was inspected and retained, not bypassed.

The native same-VM resize began at 09:18:21.794 UTC, then separately reconciled
deallocation, applied D4s_v5, and restarted the same VM. The product confirmed
finished with preserved disk/NIC/identity digest
`57fdc161fa0ed0bc2d0f626f14b6af7509b24efda0516b3123ecd38d82211dba`.
The private IP remains `10.246.1.19`; there is no public IP. Post-boot readiness
is being explicitly refreshed. No migration or network fault has started.

Post-resize GUI readiness at 09:21:40.678 UTC passed on boot
`3c50ec8b-3ca1-4997-907c-0ed6e2c013dc`, unchanged pinned CLI/commit/hash,
idle guest, 3.5114% storage, zero swap/OOM. The target FQDN matches the R2
waiter's exact binding and its ARM state is Ready / public access Disabled.
Fresh lock and activity checks found no lock or unexplained scope change.

After the fresh cloud and guest gates, the R2-only watcher was armed with
deadline **2026-09-16T09:38:32.437Z**. Transient service
`af-network-watch-20260916-b2c7214e.service` reported active/running,
RuntimeMaxSec 960, memory 256 MiB and no swap. Guest admission verified the
new boot, exact helper hashes, no previous migration, no active operation,
storage/swap/OOM gates and exact source DNS. This starts no migration itself.
The independent 45-second rule-removal timer is created only immediately before
any eventual fault. No fault is claimed merely from arming the watcher.

The installed GUI passed migration preflight and reviewed the same new target
and 5,600,000-row inventory. The previously authorized new-load step was accepted;
source credentials must still be provided through the protected GUI. Do not
submit after the watcher deadline without inspecting its retained state and
rechecking admission; never attach to a pre-existing load or overwrite evidence.

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

At the next fresh installed-GUI check (07:14:54 UTC), the same boot and pinned
artifact were idle with 4.095% storage and zero swap/OOM. Read-only cloud checks
confirmed the correct subscription, D4s_v5 / Succeeded / workflow ownership,
no RG locks and only the expected readiness command writes in the new window.
The conservative reserve and unchanged per-run deadline still admit the run.

The two reviewed helper files were placed only in this workflow's private
guest `qualification-network-tools` directory; guest byte hashes matched local:

- Observer: `29e255bcda941d1c871db2fe9daf68e97cc318bf52520261e552e57b9d59be73`.
- Waiter: `cf4eba9306284bff1f6dfb4c8e7194554c7b6d22319f6ad5297c8a5bfc8e8983`.

Transient unit `af-network-watch-20260916-8a9ae99e.service` reported active /
running with RuntimeMaxSec 960, memory 256 MiB and no swap. Its explicit wait
deadline is **2026-09-16T07:30:54.919Z**; no recurring monitor was created.
The original 45-second independent rule-removal timer remains separate from
the watcher's own lifetime. Its output/evidence are retained under this one
workflow and do not contain passwords.

The installed GUI passed new-load preflight, displayed the exact new target and
5,600,000-row inventory, and accepted the already authorized migration step.
It is now at **Read-only Neo4j source password**. The user was asked to enter
the AZ-N526 Keychain item directly and not to submit after the watcher deadline
without rechecking. No migration job or fault is claimed at this checkpoint.
After entry, inspect the selected operation and sealed fault/restoration
evidence; if the waiter expired or refused a gate, do not claim a network
recovery test or blindly rearm it against an existing/completed job.

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

### September 16 first load — no network fault injected

The user entered the protected source password. The installed GUI submitted
operation/job `ced480a1-bf31-4306-af56-ce3c0bccc525` at
`2026-09-16T07:18:53.824Z`. Guest execution ran from
`07:19:00.176920875Z` to `07:24:52.602400022Z` and finished with exit 0.
It used the unchanged pinned loader, boot
`8c9dcd46-c99f-4dd5-8966-4cf737d2e485`, configuration SHA-256
`1d9121fc4e5353c1c6e9eccebebe9bfcf505859890b5397ddd7af3ed2790fe4e`,
generation 1 and fingerprint
`18e99afaba9ac69155dc66e99ed69721f53f5ac48242fa105dcd8d49e4ed93dc`.

The independent watcher recorded 1,085,000 committed rows at
`07:22:00.922562Z`, zero rejects, checkpoint age 0.087 seconds, disk 4.071%,
memory 31,956,992 bytes and zero swap/OOM, then exited with `ValueError`.
Its original outer handler retained no stack location, so the exact failing
input cannot be reconstructed from that log. No fault threshold was reached
by the observer. An independent guest read found no before/applied/restored
fault files and no `af-network-` OUTPUT rule (iptables read exit 0).
No network interruption, restoration or recovery success is claimed.
The migration was left uninterrupted; no fault was injected into its now
completed graph and no load was replayed.

Retained hashes:

- Watcher log: `27a7795b23d1b201064081c0709646989adcbb438f24bf5d3fe9712ce9cd064b`.
- Armed evidence: `899a5d2b84c57ed8123932211736166cbfe6df325399b3163f3383c8b82c6a0a`.
- Selected job: `26e0f440a0ad47e9a6895509eb8673a8286fa5dc0e0c15385459af335c80aa06`.
- Finished state: `9000d9fc2c1f7a13fe40d283163fc21f6119b4882c95c431dbd58ce28f720ed1`.

The local observer now normalizes all 1–9 fractional timestamp digits to six
digits before Python 3.10 parsing, rather than only truncating nanoseconds.
The waiter now retains sanitized error source locations without exception
text, secrets or locals. Local observer tests are 20/20 PASS, watcher binding
tests 3/3 PASS. These are tooling regressions, not a live recovery pass.
Original deployed helper bytes/logs remain untouched. Before another live
trial, verify the fix on the guest runtime, use a fresh job/graph with reviewed
GUI binding, and recheck all gates. Do not rearm against this completed job.

An actual Python **3.10.12** guest read subsequently reproduced `ValueError`
with the original parser for fractional precision **1, 2, 4 and 5**; the fixed
normalization passed all precisions **1 through 9** on that same runtime.
This proves the compatibility defect and its fix, while the original failing
timestamp is unavailable and therefore remains an inferred cause of this exit.

Installed-GUI export/import now displays **Counts verification: PASS**.
Independent local hashing of its imported, redacted 9,619-byte report gives
`e9aebd92bdddc9f97700159dd93185fc20615d4269ea7112324d47157472c0f2`.
All 24 checks pass, with no errors/incomplete checks and exact 1,600,000
vertices / 4,000,000 edges with zero rejects. The guest's pre-redaction
`verify.json` is 11,601 bytes, SHA-256
`c4314ec44e474b3c5f4ba5fa1c0a597c9d44905b00c87364cd7305a46196fbb4`;
its different hash is expected. The committed `load.json` is 469 bytes,
SHA-256 `7e8cb82aec4deeee3722e0e73a1db06e18fce0cca48916a7d061a0d5182bf4d7`.
Full canonical digest verification was not performed for this non-faulted
attempt; neither counts success nor these observer tests close B11.

After report import, fresh ARM reads confirmed exact trial ownership and no RG
locks; the queried activity window contained only expected guest command writes.
The per-run and outer deadlines and USD 800 ceiling / USD 600 conservative
reserve remain unchanged. Cost-saving deallocation was requested only for
`af-8a9ae99ec6214a94afd1` and `af-n526-source`, and stop only for
`afpg-8a9ae99ec6214a94afd1`. Both VMs are independently confirmed deallocated;
the server is currently stopping. No resource, disk, graph, credential,
schedule or evidence was deleted. Storage charges continue, and a stopped
Flexible Server automatically starts after seven days unless acted on sooner.

The historical admission checklist below is superseded by these checkpoints;
the remaining live acceptance gate is still independent network recovery and
full canonical verification.

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
