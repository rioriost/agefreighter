# Installed-GUI recovery qualification

Status: private target and same-VM resize complete; first CSV migration and
GUI-imported exact-count verification and full canonical verification pass.
The process-fault observation window was missed and no fault was injected.
Live recovery qualification remains **not completed**.

## Candidate and preflight

Candidate source is commit `7538981cf0fc6c1bed3a50e6476861e84647a003`.
Packaging reran typechecking and all **206 extension unit tests**, all passing.
Linux/amd64 CLI and tools were built from a clean archived commit, not the
working tree. The candidate is for qualification only, not a published release.

| Artifact | Seal |
|---|---|
| VSIX | `9687d06a2ab0637bca053b1d9d9346de033696ec1597089e9439beac8d49dbad` |
| Bundled extension JavaScript | `b146dc56f466eb0851b08ead3a6173f027160caedf8e6f519e1133fbecb1aa13` |
| Linux archive | `6f10538cc70c2125cc669a4d043352efdfce510131e1bec440af3674a4a48d21` |
| Linux version | `2.4.0-dev.7538981cf0fc` |
| Linux archive bytes | 37,117,370 |

The VS Code CLI reported successful installation in the existing Mac profile.
Independent hashing confirms the installed JavaScript matches the packaged
candidate. This does **not** prove the existing Extension Host reloaded it.
GUI reload shortcuts did not change the visible state; the View menu's commands
were disabled. The user was asked to unlock/focus VS Code and reload the window.
No lock diagnosis is asserted from the disabled menu alone.

Read-only Azure checks on September 15 after 07:52 UTC confirmed all 17 trial VMs
deallocated and all 13 Flexible Servers stopped. The active subscription matches
the authorized trial. RG locks were empty. Cost query returned HTTP 429; no
immediate retry or claim of fresh actual spend was made. Retain the existing
USD 800 ceiling, USD 400 conservative accrued/noncompute reserve, and deadline
`2026-09-16T07:14:35.311Z`. Do not use the original September 9 deadline still
present in the historical `work/vscode-live-20260905-a/run.json` as renewed
authority. Storage and Cosmos can accrue charges while compute is stopped.

Recent activity included storage writes, Defender for Storage and Event Grid
configuration plus policy audit events. No security setting was changed in this
turn. Inspect actual selected resource settings and event ownership before any
subsequent mutation; do not infer that an old network exception still holds.

## Reviewed execution order

1. Confirm the installed GUI is running this candidate. Use a fresh CSV P1
   workflow first, avoiding a source password gate. Preserve every accepted base
   workflow, graph and failed-operation directory. Select the candidate manifest
   through the existing qualification control; do not manually forge workflow
   state or upgrade a previously qualified job into a recovery trial.
2. Recheck deadline, conservative whole-remaining-window cost, live quota,
   ownership, network, selected storage security and external governance. Use
   one serial runner/target pair. Record actual SKU/rates and new workflow IDs
   before approving compute. Stop if bounded total spend cannot be established.
3. Exercise CSV folder/transfer cancellation and reconciliation before starting
   inventory. Require all approved file hashes/receipts. Close/reopen the GUI
   during assessment and reconcile its retained operation without another PUT.
4. Complete inventory, private PG18/AGE target deployment and same-VM resize.
   Verify both recovery capabilities and the pinned archive at fresh readiness.
   Save a new job/graph and start the P1 migration through the installed GUI.
5. Close/reopen during load, record the unchanged operation/job, then inject a
   bounded loader-process interruption around 25% committed rows. Immediately
   retain the exact unit/PID, checkpoint, configuration hash, fingerprint,
   generation, counts, logs and health. Do not use a broad process-kill pattern.
6. Refresh failed/interrupted state, select read-only recovery readiness and
   same-job inspection, then explicitly resume through the native control.
   Checkpoint must remain at most 15 minutes old, disk below 80%, swap/OOM zero;
   all original job/graph/generation/configuration/fingerprint bindings must
   match. Preserve the old operation and its one-use continuation claim. Lost
   replies are reconciled by GET, never another resume submission.
7. Repeat with a loader reboot around 60% only after the first continuation
   progresses and all gates pass. Retain changed boot ID, old service inactivity
   and absence of automatic resume. Explicitly continue the **same job** again.
   An unavailable unit or stale checkpoint is a failed gate, not permission to
   clear a lease, fabricate status, or weaken admission.
8. Close/reopen during verification. Require complete counts with zero rejects,
   all 64 canonical ranges and the frozen P1 root, independently recomputed
   after report transfer. Capture GUI acceptance and immutable operation lineage.
9. Run an independent network-source recovery job with a narrowly scoped,
   reversible connection interruption after reviewing its exact destination and
   automatic restoration. CSV success alone does not qualify source rediscovery
   or source credential handling. Keep this case open until actual evidence exists.
10. Seal results, stop the owned test compute, and update the branch ledger.
    Do not mark B10/B11 or the extension release-qualified from preparation,
    mocked tests, load completion alone or counts-only success.

For a process termination or reboot, observe the durable checkpoint immediately
before and after the fault; the thresholds above are scheduling targets, not
claims of precise fault timing. If a fast P1 load passes a target before a safe
fault can be applied, preserve its result and use a new approved job. Do not
damage a completed generation to manufacture recovery evidence.

## GUI preparation after user reload

The user confirmed Reload Window. At approximately 08:00 UTC the installed
command palette successfully opened a workspace-free guided migration. The new
draft is `1b2c7189-b771-41e6-9086-b26029c9d8a1`: CSV/local, authorized trial
subscription/RG, Japan East zone 1, B2s_v2 and the existing `runner` subnet.
The source editor displays `csv-recovery-p1-r1`, namespace `p1`, null marker
`\N`. A local metadata read confirms 18 selected portable P1 CSV files and no
storage, assessment, guest command or migration. No Linux artifact is selected
in this draft yet; use the sealed candidate above when preparing the runner.

Installed-GUI observations for B08/B09:

- Cancelling initial file selection returned to the wizard without starting an
  upload, assessment or deployment.
- A folder without direct CSV files was rejected. Selecting the portable P1
  folder then retained all 18 CSV files. The earlier error text remained visible
  after successful selection: a presentation defect to correct/retest, not a
  failure of the retained selection or a completed transfer test.
- Cancelling the native storage/role confirmation returned `Not prepared`.
  Independent ARM GET returned ResourceNotFound for the proposed account;
  workflow metadata contains no storage intent or guest operation.
- Reopening the same confirmation retained proposed account
  `af1b2c7189b77141e69086b2`. The dialog is awaiting action-time approval to
  create it and grant the signed-in user Storage Blob Data Contributor on this
  new account only. No public source exposure or anonymous/shared-key access is
  requested. No create/role request has been submitted.

Read-only quota: DSv5 92/100, regional cores 94/101 before the new runner. Recheck
capacity at deployment/resize and include target quota separately. These checks
and cancellation observations do not close the remaining live transfer,
assessment-reload, interruption/recovery or canonical verification cases.

## Storage and CSV transfer complete; VM approval pending

After the user's action-time approval, the installed GUI submitted storage
deployment `af1b2c7189b77141e69086b2-transfer`; ARM returned Succeeded.
Account ownership/workflow tags match. The sole returned role assignment grants
the signed-in user Storage Blob Data Contributor at this account's exact scope.
Azure had changed networking to Disabled. Under the previously authorized
trial-storage-only exception, `SecurityControl=Ignore` and the unchanged
September 16 deadline were merged on this account only; network access was
restored to Enabled. Anonymous/shared-key access remain false, HTTPS-only and
TLS 1.2 remain enforced. Anonymous container-list HTTPS returned 409. No source
firewall, VNet, existing qualified target or RG-wide exception was changed.

The installed GUI reconciled storage as ready/Enabled. CSV upload confirmation
displayed 18 files and 1,168,576,671 bytes at the exact new account endpoint.
Cancelling this confirmation left zero transfer records, no assessment and no
migration. On the subsequent approved upload, all 18 files reached `uploaded`,
with the same total bytes. This is authenticated desktop-to-storage transfer,
**not** Linux import/hash verification or graph qualification.

The qualification control selected and locally verified the candidate manifest
from the seal table, then uploaded the unchanged 37,117,370-byte Linux archive.
Workflow metadata now has `developmentUpload: ready` and the exact candidate
version/hash. GUI reconnection and prerequisites generated a new VM preview:
`af-1b2c7189b77141e69086`, Japan East zone 1, Standard_B2s_v2, compute
USD 0.109/hour plus disk/network and other charges. Existing network/quota checks
passed. Prior compute remains stopped and no locks were returned. The previous
conservative reserve/deadline is unchanged; target capacity/cost must be checked
separately after complete inventory.

The final native `Create reviewed runner` dialog is pending action-time approval
for installation of this unpublished build and a new VM identity's Blob Reader
grant on this workflow container only. No public IP or source firewall change
is proposed. Metadata is `previewed`, with no submission, guest command or
migration. No Linux CSV imports, inventory, target or fault test have started.
The 18 mapping rows still need to be reviewed/entered before inventory; GUI-only
name/namespace edits are not a sealed source configuration.

## Approved runner and reviewed mappings

Following the user's approval, the installed GUI submitted the reviewed runner
once. ARM reported Succeeded at `2026-09-15T08:22:59.585858Z`; an independent VM
read confirmed Standard_B2s_v2, zone 1, private IP only and running. Its managed
identity has Storage Blob Data Reader on this workflow's container only.
The GUI reconciled provisioned state and separately verified Linux readiness at
`2026-09-15T08:30:21.496Z`: the exact candidate commit/archive above, both
`resume-inspection-v1` and `explicit-resume-v1`, idle=true, storage 3.4791%,
swap=0 and OOM=0. This does not establish actual recovery behavior.

All nine vertex and nine edge mappings were entered and reviewed in the GUI.
An independent comparison of the retained generated configuration against the
portable P1 schema passed: file selection, null marker, IDs, endpoint labels and
fields, all properties and normalized property types match. String types were
declared explicitly where the reference relies on the default. No workflow
metadata was manually edited.

The GUI imported and reconciled CARRIED_BY.csv with a matching 70,714,348-byte
full SHA-256 seal and began the next file. The remaining imports must all become
verified before complete inventory. No target or migration has started, and no
fault/recovery or canonical acceptance is claimed.

While imports were pending, the stale folder-error presentation defect was
fixed in source: a successful host initialization clears the transient error,
without clearing retained failed-operation receipts or bypassing review gates.
Typecheck and all **207 extension unit tests pass**, including a regression for
failed-folder followed by successful file selection. This UI-only change is
not installed during the ongoing pinned-candidate qualification; installed-GUI
retest remains pending.

At 08:39 UTC, four CSV receipts were GUI-verified and the fifth import had been
submitted. An expired guest-readiness gate rejected the fourth import before
dispatch; a new read-only readiness check passed at 08:37 UTC (same boot and
candidate, disk 3.8959%, swap/OOM zero), after which the explicit import succeeded.
No receipt or job state was manually repaired. A safety shutdown for this new
VM only is enabled at 07:00 UTC, before the unchanged September 16 deadline.
The target is still absent for this workflow. The new VM NIC remains private on
the selected runner subnet; transfer storage remains authenticated-only.

### CSV midpoint checkpoint

At 08:55 UTC, nine of eighteen imports are GUI-reconciled as `verified`:
CARRIED_BY, CONTAINS, Carrier, Customer, DESTINED_FOR, FULFILLS, Facility,
INCLUDED_IN and Location. The remaining nine are `uploaded`, not imported.
No import is failed/interrupted or currently outstanding at this checkpoint.
Closing/reopening the source editor during the FULFILLS import preserved the
18 mappings and the retained operation; later reconciliation completed its
receipt. This is CSV-transfer reconnection evidence only, not assessment/load/
verification phase qualification for B10.

The most recent readiness refresh at 08:50 UTC retained the same boot/candidate,
disk use 4.2686% and zero swap/OOM. A further stale-readiness rejection before
Location import was resolved by that explicit refresh, without changing file
identity. Before the next import, refresh readiness if expired. The repeated
per-file approval, command reconciliation and five-minute readiness refresh
are an observed usability limitation; a future reviewed improvement should
retain all hash, health, ownership and no-replay gates, not remove them.

The next step is to import/seal the remaining files, review the saved mappings,
then start complete inventory. Do not mark migration/recovery or P1 canonical
qualification complete from these transfer receipts.

### Managed command capacity reconciled

After the tenth verified file (Lot), Azure's per-VM limit of 25 managed Run
Commands blocked the next dispatch. No additional import was submitted by that
rejection. All 24 older completed commands were independently retrieved with
their definitions and instance views, bound to verified CSV receipts or the
matching idle candidate readiness, and archived before exact-resource deletion.
Archive SHA-256:
`3a0b62272c76effe4d3c38c7815a2d12ad574f9f9fb5d47895b410633ea6a97d`.
The retained archive is private trial evidence, not a checked-in credential or
raw cloud-response file. The controller's current command was excluded; a fresh
ARM list confirmed it was the sole remaining command. CSVs, guest operation
directories/logs, VM disks and every previously accepted graph were untouched.

At 09:10 UTC a new GUI readiness check passed on the same boot and sealed
candidate: idle=true, disk 4.4494%, swap/OOM zero. The GUI then submitted
ORIGINATES_AT import. This unblocks this trial, but manual command archival is
an observed extension usability gap, not a scalable end-user transfer flow.
No command-capacity or freshness gate was weakened; a production lifecycle
improvement must preserve evidence, outstanding references and no-replay rules.

### All CSV imports sealed

At approximately 09:28 UTC all **18/18** CSV transfers are GUI-reconciled as
`verified`, totaling **1,168,576,671 bytes**. No failed, interrupted, unknown or
submitted import remains. The same reviewed schema and original selected-file
identities are retained. Repeated explicit readiness refreshes passed on the
same boot/candidate. These receipts prove local-to-storage-to-Linux byte
integrity, not source inventory, migration, recovery or canonical graph equality.

At 09:29 UTC computer use explicitly reported that the Mac was locked and
automatic unlock was unavailable. No further GUI actions were attempted.
A read-only guest readiness command submitted at `09:28:54.999Z` is retained
for GET-only reconciliation after manual unlock; do not submit it again.
Inventory and migration are still absent. ARM lists 21 command resources after
the new imports/readiness; plan another evidence-preserving capacity review
before later phases rather than allowing the 25-command cap to interrupt them.
The one CSV VM remains running under the existing safety shutdown/deadline;
the previously accepted 17 VMs and 13 Flexible Servers remain stopped in the
latest fleet observation. The next operator action is manual Mac unlock, then
reconcile readiness, review mappings and explicitly start complete inventory.

### Inventory started after manual unlock

At 10:51 UTC the user unlocked the Mac. The old readiness request was reconciled
by GET, without replay. A separate fresh check at `10:51:56.837Z` passed on the
same boot/candidate: idle, disk 5.2846%, no swap/OOM. Fleet observation again
found only the new B2s_v2 runner active; 17 older VMs and 13 Flexible Servers
were stopped. Locks and the filtered recent governance activity list were empty.
Cost Management returned 429 again, so fresh actual spend is unavailable; the
USD 400 conservative accrued/noncompute reserve, USD 800 cap and original
renewed deadline remain unchanged.

The 21 older completed command resources were separately archived, independently
revalidated and removed, excluding the current readiness reference. Archive
SHA-256: `530a29c03725821fe7bacd2564011054d8164251964976a954fabf1a820a4f10`.
Only that readiness command remained after cleanup. This preserves all guest
files, operation evidence and accepted graphs; it does not resolve the product
lifecycle gap recorded in the remaining-validation ledger.

The reviewed installed GUI explicitly started complete inventory at
`2026-09-15T10:54:51.569Z`, operation
`c0480a91-78c1-42b9-9462-c1902029ffdd`, with source configuration SHA-256
`d2a41048183cf4a4d8b9efb21a38086a7c812b09c210771f7b84902fdccccd9a`.
The source panel was closed while submission was retained and reopened from
the wizard. It displayed the same operation; explicit refresh reconciled it as
accepted. Independent ARM listing showed one inventory submission, not a
duplicate. This establishes source-panel reconnection during assessment, not
an Extension Host crash/reload test or the remaining load/verify cases.

The complete inventory report was generated at `10:55:18.177622407Z`, exported
once and GUI-imported with its 3,219-byte SHA-256 seal
`9f955488357ebe2ec05c7f29e209dd1eb3045faacb12765cbd2e53f2146c1d86`.
An independent local recomputation matched. Outcome/read-only/source-counts/
source-unchanged all pass, errors and incomplete checks are empty. Exact totals
are **1,600,000 vertices + 4,000,000 edges**, all 18 mapped labels accounted for.
Recommended target storage is an estimate of 9,848,627,370–27,719,516,636 bytes,
not measured final database size or migration verification.

### Private target preflight / action-time approval

An initial target review was rejected before any Azure write because readiness
had expired. A new GUI check at `11:00:37.797Z` passed (same boot/candidate,
idle, disk 5.2859%, no swap/OOM). Re-entering the reviewed choices passed the
live target preflight. The native final dialog proposes new private server
`afpg-1b2c7189b77141e69086`, Japan East zone 1, PostgreSQL 18/AGE,
Standard_D4ds_v5 with 128 GiB, single-server trial, new non-overlapping delegated
subnet `10.246.18.0/24` and private DNS in the existing VNet. No public access or
peering. Same-runner resize to D4s_v5 remains a separate later action.

The displayed combined compute rate is USD 0.736/hour, plus the unchanged
USD 400 accrued/noncompute reserve, within USD 800 through
`2026-09-16T07:14:35.311Z` (roughly USD 415 conservative envelope at review,
not a bill or automatic shutdown guarantee). Independent regional quota read:
DSv5 92/100, regional 96/101 before resize. Existing shutdown remains 07:00 UTC.

The final `Save plan and approve target deployment` dialog awaits action-time
confirmation for the new database administrator credentials stored by the
extension in SecretStorage. No target intent/deployment, credentials or output
files have been created at this boundary. After approval, select a fresh local
output folder, reconcile the exact deployment, and continue the reviewed
same-VM recovery qualification. Do not mark this as migration completion.

### Approved private target submitted

The user approved the exact target/SecretStorage scope. Since the previous
preview had expired during the wait, it was cancelled before creating files or
resources; fresh readiness at `11:20:17.549Z` passed (same boot/candidate, disk
5.2861%, idle, no swap/OOM). Re-entered choices and live preflight matched the
approved sizing, network, price, reserve, deadline and ceiling. Locks and
filtered recent governance activity remained empty.

The native GUI saved a create-only LoadJob and target plan into the private
local `work/csv-recovery-p1-r1` directory. Plan hash:
`cd6cb86bf324759a674c8900cc99f9d23ba73cae80c8f8c4adbd073358a2058b`.
The GUI retained `submitted` intent and ARM deployment
`afpg-1b2c7189b77141e69086` was Running at `11:22:39.136164Z`.
No credentials were displayed or copied into this report. Target creation is
not yet proof of AGE readiness or migration success; reconcile by the exact
deployment ID without resubmission.

### Private target provisioned; GUI capture interruption before resize

The installed GUI reconciled the target as `provisioned`. Independent ARM
operation reads show successful server, database, extension allowlist, preload,
delegated subnet, private DNS and VNet link creation. The server is PostgreSQL
18, Standard_D4ds_v5, Japan East zone 1, with public access Disabled.

Through the installed native execution control, the approved preload restart
was submitted at `2026-09-15T11:29:52.677Z`. Subsequent independent Azure reads
show Ready, `azure.extensions=AGE`,
`shared_preload_libraries=pg_stat_statements,age`, and no pending restart for
either parameter. This proves service configuration, not successful SQL AGE
preparation or migration. Local `targetRestart` still says `submitted`; the
next GUI action must reconcile it read-only, not submit another restart.

Fresh GUI guest readiness at `11:30:13.738Z` passed with the same pinned build
and boot, idle=true, disk 5.2866%, no swap/OOM. Immediately before these actions,
RG locks and filtered recent delete/lock/policy activity were empty. The USD 800
ceiling, USD 400 reserve and September 16 deadline are unchanged.

Before selecting the resize action, native screen observation failed twice
with ScreenCaptureKit error -3811. App inventory remained available, but it
does not establish that the screen is unlocked or usable. No lock cause is
asserted, no blind GUI input was sent, and no workflow metadata was edited to
bypass the GUI. The user was asked to restore the visible VS Code session.
Independent reads confirm the runner is still B2s_v2/running and local resize
and migration intents are absent. Both new compute resources remain retained
within the approved window; the existing VM-only shutdown does not stop the
Flexible Server. Resume with GUI preload reconciliation, fresh readiness and
the reviewed same-VM resize sequence before creating the new migration job.

### GUI restored, resize complete and first migration submitted

The user reported that the Mac was not locked. Reinitializing the UI connection
restored VS Code capture. The native control reconciled preload as `finished`.
The same-VM deallocate/resize/start sequence began at `11:33:18.951Z` and reached
`finished` without changing its retained identity seal:
`58c3f0d1181312faa13861b522785464ce88488659287c17a7fb71428855a560`.
Readiness at `11:36:47.784Z` confirmed D4s_v5's new boot
`13faacc0-4f69-490b-9afc-853f75d34b45`, the original pinned candidate, idle=true,
disk 5.2873%, no swap/OOM and both recovery capabilities. Recent filtered
governance and locks were empty; budget/deadline were not renewed.

The installed GUI approved and submitted the new CSV migration at
`2026-09-15T11:38:02.279Z`, job/operation
`ea740a86-daaf-42b8-b92d-709312c9e067`. Retained guest configuration SHA-256 is
`f06f545e6eec28752488415015c670b9b54933cde2d527bc3717a896845028e8`.
Closing and reopening the wizard, selecting the original saved CSV workflow,
and refreshing retained migration preserved that operation/job; no second load
was submitted. This is load-phase panel-reconnection evidence, not an
Extension Host crash/reload or completed B10/B11 qualification.

### Process-fault window missed safely

The new guest-only observer defaults to read-only. Its explicit P1 SIGTERM mode
checks exact workflow/operation/job/configuration/boot and the load child,
checkpoint age, rejects, memory, storage and swap/OOM. It seals evidence before
signalling a PID-bound descriptor, refuses replay and stops at the reviewed
upper window. Local admission tests cover unsafe/missing gates; they do not
prove a live interruption.

The first observer failed before signalling with a timestamp parser ValueError
on the older guest Python. Nanosecond-to-microsecond normalization was added,
with a regression test (five local tests pass). The second observed
**3,815,000 committed rows** at `2026-09-15T11:41:29.327877Z`, generation 1,
the retained fingerprint, zero source/target rejects, checkpoint age 0.883s,
disk 5.2931%, zero swap/OOM and about 1.223 GB cgroup memory. Because it was
already beyond the 2.5-million-row upper bound, it refused SIGTERM. No fault or
resume success is claimed. Preserve this run through verification; do not
modify a completed generation to manufacture recovery evidence. Observer
process matching was also corrected to compare resolved executable paths.

VS Code capture then failed again with -3811 while Finder remained readable;
reinitializing the connection did not immediately resolve it. The user was
asked to bring VS Code forward, not to unlock an already unlocked Mac. Azure
read-only observation remains available. No invisible UI input or controller
state bypass is allowed while capture is unavailable.

### First load and imported counts verification complete

GUI capture subsequently recovered. The guest completed the original job at
`2026-09-15T11:43:13.281292701Z`, after starting at
`11:38:29.956584016Z`: approximately 4 minutes 43 seconds for preparation, load
and counts verification, excluding provisioning, inventory and full digest.
All **5,600,000 rows** committed with zero rejects. The installed GUI reconciled
the original operation, transferred the report and displayed counts PASS.
Independent local hashing confirms the 9,619-byte report SHA-256:
`9d1eb718b4227fe289d29458adffd9ca83239e226ea09d98dc8913714c24417d`.
All 18 label counts agree; 24 checks pass with no errors or incomplete checks.
This is counts acceptance, not full property equality or recovery acceptance.

The guest's fault-evidence file is absent: no process fault, reboot or resume
occurred. Preserve this completed generation as an additional clean trial.
Any actual interruption trial requires a new reviewed job; do not resume or
damage this completed job. The observer now checks children of all Go worker
threads, with six local unit tests passing. Its earlier empty PID observation
has not been causally reproduced or live-qualified; neither the resolved-path
comparison nor thread enumeration is evidence of a successful fault injection.

Fresh idle readiness at `11:47:13.356Z` retained the pinned candidate and boot,
disk 5.3198%, swap/OOM zero. The user approved the independent full P1 verifier
for this exact job: commit `8a23a5109798ec906109532e4cc6c32308b3c824`, archive
SHA-256 `60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d`,
7,248,083 bytes, read-only target checks, maximum 25 minutes / 4 GiB.
The expired readiness was refreshed before dispatch: `11:52:53.472Z`, idle,
5.3203% disk, no swap/OOM. Locks were empty; recent policy events were audits,
not a detected enforcement mutation. Only this runner/target pair was running.
Budget, deadline, graph and credential scope are unchanged.

The native GUI submitted full verification once at `11:54:40.390Z`, operation
`6d5b62a4-7f7a-45a0-9ad1-03a6eb9125e5`, bound to the original migration job.
Azure independently reports Running, guest start `11:54:58Z`. Reconcile that
exact operation and its export; do not replay verification. Canonical acceptance
is still pending, and this does not turn the clean load into a recovery test.

While ARM still reported the verifier Running, the migration wizard tab was
closed, reopened through the command palette and reconnected to the original
saved CSV workflow. It retained the same verifier and migration operation IDs.
This exercises panel reconnection during independent verification; it is not
an Extension Host termination/reload or automatic recovery test. The retained
command must finish and its report be reconciled without a second submission.

### Full canonical qualification PASS; compute stopped

The independent verifier finished successfully at `2026-09-15T11:57:03Z`
(Azure start `11:54:58Z`, exit 0). The installed GUI reconciled the original
operation, exported the exact report through command
`667890a6-a0af-429d-8384-7345a21683fc`, imported it and displayed
**P1 full canonical digest: PASS**. The retained report is 23,223 bytes,
SHA-256 `a649dbecc3f0db163c2f4f6292b2a8b6b666bff6b454ff6fe955b5376d4dac6e`.
An independent Mac-side check verified report bytes/hash, original job binding,
read-only outcome, all **64 matching leaves**, and all **5,600,000 records**.
Recomputing both roots from the leaf fields produced the frozen canonical root
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
No second verifier submission was used after panel reconnection.

Post-verification guest observation confirms no active workflow marker, disk
7% (`df` rounded), zero swap/OOM, and no SIGTERM evidence file. This closes the
additional clean CSV trial only, **not B11**. B10 has panel-close/reconnect
evidence across inventory/load/independent verification; actual host crash/reload
and counts-verifier interruption remain separate gaps.

After verifying exact workflow ownership and empty RG locks, deallocation of
the trial VM and stopping its Flexible Server were submitted. No resource,
graph, disk, report or failed-run evidence was deleted. Subsequent independent
Azure reads confirm `PowerState/deallocated` and Flexible Server `Stopped`.
Flexible Server warns it automatically starts after seven days. Retained
storage/Cosmos charges are not eliminated by compute shutdown.

### Second CSV recovery draft (not deployed)

The user requested continuation after the clean trial. Fresh reads at about
12:04 UTC confirm all trial VMs deallocated, all Flexible Servers stopped,
the authorized subscription selected, empty RG locks, and no recent delete,
deny or deploy-if-not-exists activity in the bounded check. The USD 800 ceiling,
USD 400 reserve and `2026-09-16T07:14:35.311Z` deadline remain unchanged.

The installed GUI opened a new CSV/local workflow
`54da6ddd-27d2-45e0-bb68-cf5f352801db`, using the existing trial group, Japan East
zone 1, B2s_v2 and the original runner subnet. The source editor displays
`csv-recovery-p1-r2`, namespace `p1`, null marker `\N`, and all 18 original
portable P1 files. Mappings still require GUI review. This is a new draft, not
a modification of the prior accepted generation or workflow metadata.

The native dialog awaits action-time approval to create storage account
`af54da6ddd27d245e0bb68cf` and grant the signed-in user Blob Data Contributor on
that account only. Independent Azure GET returns ResourceNotFound. No storage
deployment, role grant, runner start, inventory or migration was submitted.

For this attempt, validate the observer before starting the migration and start
its bounded observation immediately after the GUI creates the exact job ID.
Do not spend the short P1 load window on panel-close tests before the observer
is watching. Require a current durable checkpoint and the exact loader child;
stop without signalling on ambiguity or after the upper boundary. Panel
reconnection already has separate evidence; it must not displace the planned
fault. Any later reboot remains a separately gated same-job step, not an
automatic consequence of starting this observer.

### Second CSV trial: transfers and reviewed mapping ready

Following the user's exact storage approval, deployment
`af54da6ddd27d245e0bb68cf-transfer` succeeded. The signed-in user received Blob
Data Contributor on that new account only (assignment
`aca68832-c107-4fc2-90f4-a9cd1cc74282`). The previously approved trial-storage-only
`SecurityControl=Ignore` exception and the unchanged expiry were applied to this
account, then public network access was restored. Anonymous access and shared
keys remain disabled, HTTPS and TLS 1.2 are required; anonymous listing returned
HTTP 409. This did not change the resource group or any source firewall.

The installed GUI uploaded all 18 portable P1 files (1,168,576,671 bytes) and
prepared the pinned development archive for commit
`7538981cf0fc6c1bed3a50e6476861e84647a003`, SHA-256
`6f10538cc70c2125cc669a4d043352efdfce510131e1bec440af3674a4a48d21`.
All nine vertex and nine edge mappings were entered and reviewed in the source
form. An independent read-only comparison against `csv-source.json` passed for
labels, identities, endpoints, property types, defaults and selected-file
bindings, normalizing implicit string types. No workflow state was edited
outside the extension.

At 12:24 UTC the GUI prerequisite preview selected the same group/subnet,
Japan East zone 1, `Standard_B2s_v2`, compute estimate USD 0.109/hour. Bounded
governance checks found no RG locks or recent delete/deny/deploy-if-not-exists
events. The final VM dialog is awaiting action-time approval for creation of
`af-54da6ddd27d245e0bb68`, execution of the pinned development build, and its
identity's Blob Reader grant on this workflow container only. No public IP,
SSH ingress, peering or source firewall changes are proposed. No VM, target,
guest import, assessment, migration or recovery fault has started for this draft.
The USD 800 ceiling, USD 400 conservative reserve, and hard deadline remain
unchanged. The first trial's canonical PASS remains a clean qualification;
B11 is still open.

The user subsequently approved the exact VM/build/container grant. The installed
GUI submitted deployment `af-54da6ddd27d245e0bb68` once; ARM independently
reported `Running` with timestamp `2026-09-15T12:26:56.368496Z`. All other 18
trial VMs and 14 Flexible Servers were confirmed stopped before this submission.
VM readiness and the 18 Linux imports are still pending. The six observer unit
tests passed again; this is not live fault-injection evidence.

The new VM is running privately in zone 1 on the reviewed B2s_v2. Independent
RBAC inspection confirms its sole returned grant is Blob Data Reader on the
exact workflow container. A 07:00 UTC safety shutdown was enabled for this VM
only, before the unchanged hard deadline.

The first readiness request raced bootstrap: retained command
`af-c2a15bb6-6ff3-42c7-aec6-b7b5ca4431ff` exited 127 because
`/usr/local/bin/agefreighter-tools` was not yet present. Read-only guest evidence
subsequently showed cloud-init finished at `12:28:38Z` with no errors and the
binary present. No bootstrap rerun, alternate download or safety bypass was
used. A new GUI readiness check at `12:29:38.430Z` passed with the exact pinned
candidate, boot `489e1229-d666-4895-82bd-46dce8916445`, idle=true, disk 3.5086%,
swap/OOM zero and both recovery capabilities. The first CSV import was then
submitted through the GUI. ARM provisioning success alone was not treated as
guest readiness. This exposes a startup-status UX gap for B09, not a recovered
migration or completed bootstrap-failure qualification.

### Second trial: nine Linux receipts and command capacity reconciled

The installed GUI verified nine imports: CARRIED_BY, CONTAINS, Carrier,
Customer, DESTINED_FOR, FULFILLS, Facility, INCLUDED_IN and Location. Each
retained receipt matches the original file byte count and full SHA-256. Lot's
first confirmation was rejected before submission because readiness expired;
there was no failed import or partial generation to resume.

Fresh GUI readiness at `12:55:03.237Z` retained the same boot and pinned build,
idle state, disk below 80%, and zero swap/OOM. Fresh RG locks and bounded
delete/deny/deploy-if-not-exists activity checks were empty. At that point 24
ARM command resources occupied this VM's 25-command limit. Twenty-two successful
commands were independently fetched and validated against the nine verified
CSV operations or the matching idle readiness profile, then archived before
exact-resource deletion. Latest readiness and the initial bootstrap-race failure
were excluded. Archive SHA-256:
`8675610513e07d10e018a454872cc7de4500dd0f527e563597472ff27aa8cc2c`.
Raw definitions/results remain in private local trial evidence. Immediate
readback confirmed precisely the two excluded commands remained in ARM. No CSV,
operation directory, guest log, disk or graph was removed. This manual procedure
remains an end-user lifecycle gap; it is not an implemented extension feature.

The GUI then explicitly submitted Lot's first actual import. The remaining
imports, complete inventory, private target, migration faults and same-job
recovery still require execution. Neither B10 nor B11 is closed by this work.

### Second trial: all CSV imports verified; complete inventory approved

All **18** CSV imports are now GUI-reconciled as `verified` (1,168,576,671
bytes). Independent Mac-side rehashing of all 18 original files matches every
retained byte count and SHA-256; the nine vertex and nine edge mappings still
match the frozen portable P1 schema. No failed/interrupted or pending CSV import
remains. Linux readiness at `13:25:32.449Z` retained the exact boot/build,
idle=true, disk 5.3148%, zero swap/OOM.

The VM again reached 25 ARM command resources after transfer and readiness
checks. A second archive retained 23 independently validated successful command
definitions/results before deleting only those exact ARM resources. Its SHA-256
is `fcace698846ce714794ff67d8824bef8a4b4f981421a2e3cb88830c4e3bb5400`.
The first archive was not overwritten. ARM readback again confirms only latest
readiness `af-b1feae37-6069-4faa-bf1a-40c65fbc08b0` and the initial bootstrap-race
failure remain; all guest evidence and source files are preserved. This manual
capacity intervention remains a release-readiness gap, not GUI qualification.

The source form was reviewed again and the native GUI approved a complete CSV
inventory: all mapped rows, before/after hashes, maximum 30 minutes / 4 GiB /
no swap. The extension is rechecking Linux readiness before dispatch. Target
creation, migration, injected faults and explicit same-job resumes have not
started for this workflow. Full canonical qualification remains future work.

### Second trial inventory PASS; target review paused at Mac lock

Inventory operation `5bf5cb63-4614-4755-9fcc-9149d5a8f064` was submitted at
`2026-09-15T13:28:59.107Z`. Its GUI-transferred report was generated at
`13:29:36.916674761Z`, has 3,219 bytes and SHA-256
`66394cc395f28307d90e7f34ab1097aff44937fe75e4612fe33cdb40857c2821`.
The installed GUI displayed **Hash-verified source report**, and an independent
local hash check matches the receipt. Outcome is pass, errors and incomplete
checks are empty: read-only scan, complete mapped counts and before/after
file immutability all pass. Counts are **1,600,000 vertices and 4,000,000 edges**
across the expected 18 labels. The source fingerprint is
`8e518228a9faa4c5c80a0764779b6712a628df907e9bc64d09b32494b09badfb`.
This is source inventory, not target integrity or recovery qualification.

The target review inputs selected new server `afpg-54da6ddd27d245e0bb68`,
PostgreSQL 18 / AGE, D4ds_v5, 128 GiB, and the same runner resized later to
D4s_v5. Independent VNet inspection found `10.246.19.0/24` non-overlapping;
that subnet was entered without creating it. Existing Japan East/zone 1 placement
is retained. The default 24-hour deadline was replaced with the authorized
`2026-09-16T07:14:35.311Z`; ceiling USD 800 and reserve USD 400 were retained.

Before the final target preview could be inspected, computer use explicitly
reported the Mac locked and automatic unlock unsuccessful. GUI work stopped
and manual unlock was requested; no lock bypass was attempted. Independent
metadata confirms no target intent or migration exists; ARM GET for the proposed
server returns ResourceNotFound. Target price/preflight/final approval must be
rechecked after unlock, before saving a plan or creating any target resource.
The sole new runner remains B2s_v2/running with the previously configured
07:00 UTC safety shutdown. No inventory/transfer command is active. Prior
qualified targets, source files, logs and both command archives remain retained.
