# B09 / B10 bounded execution checklist — offline preparation

Prepared from current source and retained documents on September 23. **No live
action is authorized by this checklist.** No cloud, GUI, operator-store or
credential access, start/stop, process termination or installation was performed
for this review. Root alone coordinates any later approved live session.

Use the exact resource/window/budget boundaries in
[the next live scopes](b09-b10-b12-live-next-20260923.md). Candidate observations
there are historical, not a current health assertion: workflow
`ae952310-5eba-42b6-9fe3-9db00e93cdac`, runner
`af-ae9523105eba42b69fe3`, 11 controls, prior inventory
`3a4ffae1-8727-464b-8968-40bed2f42c69`, source
`afpg-p1-source-20260907`. Recheck them before a later authorized start.

## Findings that affect the next trial

1. **The minimum B09 readiness pair is supported.** Two explicit readiness
   submissions, each followed by successful reconciliation, retain two sealed
   receipts from the current boot. With no other submissions, 11 controls become
   13 before one candidate is removed. Restarting invalidates old boot evidence;
   old receipts cannot substitute for this fresh pair.
2. **The pre-target B10 candidate can retain its accepted inventory.** A finished
   assessment with a sealed report is not `assessmentActive`. An explicitly
   reviewed new inventory adds the old assessment to `assessmentHistory` before
   dispatch. History is capped at 16. A migration freezes this route, so do not
   substitute a migrated workflow or edit phases to manufacture eligibility.
3. **An active remote worker does not imply a held local lock.** The automatic
   watcher locks only around each control step and releases while sleeping.
   A genuine active-inventory host crash may leave no local lock. Record the
   observed boundary; never create a lock to join separate evidence artificially.
4. **Persisted `running` is not independent proof of a live worker.** Guest
   `Manager.Status` reads retained state and checks boot, but does not inspect
   the process/service. Correlate a fresh running receipt with narrowly approved
   guest process evidence for the exact operation unit before the host crash.
5. **Deployment modal consent binding needs correction before its stale/duplicate
   live test.** At this review, `runnerMigration` checked the window's hash before
   showing the modal, then passed the latest disk record to `submitRunner` without
   comparing it to the reviewed snapshot/expiry. Another window can renew a
   preview, including its expiry. Validating the latest preview does not validate
   the original consent. Root/supervisor were notified; require a captured review
   binding, original expiry and action-time trust/development-opt-in checks plus
   deterministic regression before attempting live confirmation. This checklist
   makes no production change and does not claim that finding has been fixed.

Source anchors: `core/runnerAssessment.ts` (`assessmentActive`, `startAssessment`),
`core/runnerGuest.ts` (`dispatchGuest`, `reconcileGuest`), `runnerWatch.ts`,
`internal/runner/manager.go` (`Status`, `unit`), `runnerMigration.ts` (`deploy`).

## B09: smallest useful running-control lifecycle

Source and all targets stay stopped. Prepare the fresh 60-minute maximum runner
window, USD800 cumulative ceiling, monitor and final five-minute stop reserve
before the first start. Stop on completion, terminal failure, 15-minute idle
input wait or deadline. No time extension is implicit.

1. Command Palette: **AGEFreighter: New Guided Migration**. Choose **Reconnect
   to a saved workflow**, then the exact candidate UUID. Reconnection itself
   only selects retained state; do not use deployment buttons.
2. After the separately authorized runner start, click **Check Linux guest
   readiness**, then **Refresh guest command** until that exact control has a
   successful, fully populated receipt. Pending/unknown is a reconciliation
   state, never a reason to submit another readiness control automatically.
3. Verify pinned guest commit/archive, current boot, idle health, disk below80%,
   no swap/OOM. Click **Check Linux guest readiness** once more and reconcile
   **that second control** successfully. It must be newer and same-boot. The
   current proof's freshness starts at submission, not at the latest GET.
4. Command Palette: **AGEFreighter: Review / Reconcile Readiness Control
   Removal**. Select the candidate workflow and the **older of the two new
   receipts**. The newer receipt stays referenced/current. Observe the complete
   native modal, exact command ID and receipt hash, then **Cancel**. Require no
   new archive/removal intent and no effective deletion; retain control-plane
   and local evidence separately from controller-level dispatch assertions.
5. Reopen a fresh review only while current readiness is under five minutes old.
   Obtain exact action-time approval for **Archive and remove this record**.
   Production flow rechecks ownership and both successful proofs, saves/syncs
   its v2 archive and intent, then sends at most one DELETE for the selected ID.
6. Invoke the same command and select that receipt again. It now selects
   **GET-only reconciliation**, without a new deletion modal. Require exact
   ARM404 plus retained matching archive before recording `absent`. HTTP202 or
   the first success notification alone is insufficient.

Do not remove historical Pending/Updating records, remove the latest proof,
start the source, or submit inventory during this B09-only lifecycle. If Azure
instance view lacks matching terminal output/timestamps, preserve the refusal
and stop; do not manufacture successful evidence.

### Additional bounded faults: distinguish their effects

The receipt-expiry, receipt-duplicate and lost-DELETE-reply rows below are
**optional supplemental removal variants**, not extra mandatory cases beyond
native lifecycle Cancel, one approved DELETE and GET404. The original B09
expired/duplicate **runner preview** and lost **deployment** reply requirements
remain distinct; removal variants cannot substitute for them.

| Case | Lowest-risk defined mechanism | Honest acceptance / limits |
| --- | --- | --- |
| Receipt-removal expired review | Open a valid native removal modal, let its original current-readiness deadline pass, then confirm the stale review only within the explicitly authorized negative-test scope | Reject before archive/intent/DELETE. Use real elapsed time; never alter system clock or saved timestamps. A later lifecycle attempt needs one separately requested fresh readiness control. |
| Receipt-removal duplicate windows | Two windows **sharing the same profile/store** review the same eligible receipt. Complete one exact approved removal/intent, verify it is durable, then confirm the other already-open stale modal | Second path must refuse its changed record before another DELETE/archive/intent. Do not use separate copied stores: their independent locks cannot prove cross-window exclusion. This is removal duplication, not runner-deployment duplication. |
| Runner-deployment stale/duplicate preview | First fix the consent-binding issue above; then use an eligible unsubmitted preview whose pinned artifact is already valid, with no first deployment authorized | Waiting beyond the original expiry and/or a second window renewing its preview can demonstrate stale consent refusal after the fix. The deployed ae952310 candidate is ineligible for this preview path. Current missing public2.4.0 artifact also prevents an ordinary fresh preview; do not bypass it. |
| Lost readiness DELETE reply | Prefer retained real transport uncertainty if it occurs: preserve `unknown`, same intent and archive, then reconcile the exact resource by GET only | Existing injected local lost-ACK tests are not a real Azure reply loss. Deliberate deterministic loss requires a separately reviewed, exact transport mechanism. Host/network-wide outages or arbitrary proxy/certificate changes are inappropriate. |
| Lost runner deployment reply | Requires an actual separately approved fresh deployment and proof of server acceptance/client uncertainty, then same deployment-ID reconciliation | Existing runner restart/removal authorization does not authorize a new VM/deployment. Synthetic transport tests and B08 committed CSV-upload ACK loss cannot close this branch. No replay or local phase rewriting. |

## B10: one real active signed-in inventory crash

The remaining gap is **one** active signed-in Azure crash. Do not repeat three
crashes merely because assessment/load/verifier Reload Window are separate
historically accepted cases. Native unsigned-in lock recovery and the isolated
SIGKILL harness are already separate accepted local evidence.

1. Obtain the additional exact crash/source-start/read-only inventory/report
   transfer scope and bounded deadline. Prepare the source credential privately
   before starting compute. If sharing the B09 window, leave enough remaining
   time for the 30-minute guest cap, bounded export and final stop reserve;
   otherwise defer to a separately authorized window rather than extend time.
2. Reconnect to the exact candidate, verify current pinned guest health, and
   select **Configure source & assessment**. Inspect all18 retained mappings;
   **Review source settings**, then **Approve complete source inventory** and
   native **Approve source reads** exactly once. Source/runner only are started;
   no target, migration, mapping redesign or credential reset is part of this.
3. Record new inventory operation UUID, exact ARM control ID, original and new
   configuration hashes, guest boot and submission time. Verify the previous
   accepted assessment is retained unchanged in history and its old report is
   intact. Never record protected parameters, password/environment files or
   capability URLs as evidence.
   Retain an allowlisted timeline of `guestCommand.id`, `operation`, `action`,
   `phase` and time as status controls replace the current pointer. ARM control
   count/output alone does not distinguish an inventory dispatch from a status
   request returning that inventory's state; never decode protected parameters
   to infer the distinction.
4. Establish actual worker activity: a fresh same-operation `running` receipt
   plus a bounded read of only nonsecret service properties for
   `agefreighter-assessment-<operation>.service` (for example ActiveState,
   SubState, MainPID, MemoryCurrent and invocation/start identity) and matching
   process identity through the already approved guest observation channel.
   This may itself consume a separately authorized read-only guest control; it
   is not an ARMGET-only action. No worker environment or configuration/secret
   dump is needed. If the operation finishes first, preserve it and mark the
   active-crash trial unexecuted; do not auto-repeat inventory.
5. **Before any termination**, root binds the exact approved VS Code window and
   profile to its local Extension Host using independent VS Code window/process
   information and contemporaneous logs. Cross-check PID, parent/app path and
   process start identity. If a production lock currently exists, its PID and
   same-boot identity must agree, but a lock is not required for an active-cloud
   crash. A process name or the newest `Code Helper` PID alone is insufficient.
   Recheck identity immediately before the separately approved single-host
   termination. Never terminate the main application, a renderer, the guest
   worker, or another editor/session. This checklist supplies no kill script.
6. Preserve independent main-log evidence for that exact host's forced exit and
   the post-crash operation IDs. Reopen/reconnect. If a same-boot dead-owner
   production lock remains, use **AGEFreighter: Review Interrupted Runner
   Lock**, check exact hashes/PID, and obtain native **Recover local lock**
   approval. Verify the immutable archive and unchanged inventory identity.
   If no lock remains, record that outcome instead.
7. **Configure source & assessment** restores the existing operation and starts
   its bounded watcher; do not choose the inventory approval button again.
   Pending inventory/status receipts reconcile by GET. Once a previous receipt
   is terminal, explicit/automatic status checking can create bounded read-only
   **status** controls for the same inventory operation. Distinguish these from
   a second **inventory** dispatch. Watcher steps are120seconds apart, at most15
   per invocation; a reopened watcher has a new bound, so count cumulatively.
8. Require one inventory submission, unchanged operation/configuration identity,
   a complete sealed1.6M-vertex/4M-edge/18-label report and accepted evidence
   preserved. Use **Transfer / open verified report** and native **Transfer
   verified report** for only this report. Export is one separately approved
   control plus bounded GET/import; no source re-read. Then independently
   confirm runner deallocated/source Stopped and pause the scoped monitor.

### Capacity, health and proof limits

The hard managed-control limit is25. Starting at11, the basic B09 pair/removal
nets12; adding one freshness renewal nets13. A subsequent B10 readiness,
inventory, independent guest observation, up to seven status dispatches in one
15-step watch, and report export can reach24. Extra manual refreshes, uncertain
receipts or a restarted watcher can exhaust the remaining margin. Recount before
each new dispatch and reserve status/export capacity; never remove unrelated
controls or reinterpret quota refusal to force completion.

Guest inventory service declares `Restart=no`, `MemoryMax=4G`, zero swap and
`RuntimeMaxSec=1800`. These configured limits do not by themselves prove observed
RSS or service activity. An external scoped safety monitor is still required:
the pre-target source watcher has its own30-minute bound and cannot infer the
operator's shorter compute authorization deadline.

## Finite approval inventory: decision paths, not Cartesian repetitions

This source inventory updates the September17 list. It is **not** a statement
that all rows are untested, or a reason to repeat every source/SKU combination.
Credit existing evidence per distinct effect before choosing remaining cases.
Shared-dialog alternatives below do not each add a separate Cancel requirement.

| Controller | Distinct current approval effects |
| --- | --- |
| Runner creation | Create reviewed discovery VM; stale/duplicate consent binding prerequisite above |
| Development runner | Prepare pinned archive; upgrade idle runner |
| Source assessment | PostgreSQL catalog read; catalog transfer; selected mapping adoption; retain failed assessment; Cosmos Reader grant; CSV upload; CSV import; transfer storage/user-role creation; retain rejected export; assessment report transfer; sampled source read; complete inventory |
| Target review | One target modal with Save plan only / save and deploy alternatives; separate failed-preload repair modal |
| Migration execution | Renew local cost authorization; preload restart; same-VM resize; new migration; explicit same-job resume; checkpoint inspection; archive empty-target failure; target diagnosis; migration report transfer |
| Canonical qualification | One shared full P1 verification modal for new qualification or corrected requalification; separate P1 failure diagnosis |
| Evidence lifecycle | Exact readiness-control removal; local crash-lock recovery |

Selection/file/folder/credential cancellations remain separately described by
their existing evidence. Reconciliation of an already retained operation is not
a fresh mutation approval. Shared UI code does not merge new migration and
explicit resume, or sampled reads and full inventory, into one effect.
Conversely, Save plan only is a positive alternative in the target modal, not
another Cancel modal. Corrected requalification uses the same P1 modal and the
same immediate Cancel return; its history-preservation behavior is a separate
safety test, not a reason to require another identical native cancellation.

Previously proved facts to preserve: live target quota refusal is explicitly
credited to B02/B09 in
[September15 recovery evidence](recovery-execution-20260915.md#target-preflight-safely-refused-regional-vcpu-exhaustion);
the installed valid-placement/missing2.4.0-artifact refusal is in
[September22 placement evidence](placement-arm-readonly-20260922.md#follow-up-post-fix-valid-placement-control-completed).
Their historical validity does not depend on manufacturing the same denial
again today. B08 CSV lost upload acknowledgement is not lost ARM deployment
acknowledgement. Current branch totals must change only for newly observed
specific cases, never for this offline checklist.
