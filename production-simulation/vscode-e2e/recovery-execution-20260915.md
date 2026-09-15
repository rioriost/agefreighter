# Installed-GUI recovery qualification

Status: runner provisioned and pinned Linux readiness verified; CSV guest imports
are in progress; live fault tests **not started**.

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
