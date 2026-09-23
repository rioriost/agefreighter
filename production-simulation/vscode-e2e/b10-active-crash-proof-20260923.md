# B10: one approved active-inventory Extension Host crash

Offline preparation from source, September 23, 2026. **No compute, guest
observation, local process inspection or termination was performed for this
checklist.** Root reports fresh approval for one bounded session, maximum 60
minutes from first compute start, on only:

- Workflow `ae952310-5eba-42b6-9fe3-9db00e93cdac`, runner
  `af-ae9523105eba42b69fe3`.
- Read-only PostgreSQL source `afpg-p1-source-20260907`.
- One explicitly approved new inventory, one precisely identified local
  Extension Host interruption, reconciliation and that report's import.

This supersedes the *authorization status* of the older preparation document
only to the extent of that explicit approval. It does not authorize a target,
migration, upgrade, extra inventory attempt, receipt deletion or another crash.
Root owns execution and re-observation. Preserve the final five minutes for
deallocation; stop early on terminal failure/completion or the authorized idle
input limit. Never extend the window to obtain a successful fault result.

## Prepare the race-sensitive proof before approving source reads

1. Resolve the exact normal VS Code window/profile and its local Extension Host
   PID **before** submitting the inventory. Record app executable, PID, parent
   PID, process start identity and associated window/log identity. Root must
   recheck immediately before the single interruption. Do not pick the newest
   `Code Helper`, assume parentage identifies a window, or terminate a renderer,
   main application, another profile, or a guest process.
2. Prepare the read-only guest observation payload and evidence filenames before
   submission; bind its unit name only after the new operation UUID is known.
   Prepare the reopen/reconnect navigation too. Interactive reader credentials
   stay entirely with the user; no credential inspection is part of proof.
3. Reconfirm exact resource ownership, the current count of retained managed
   controls, unchanged source mapping/configuration hash and no target/migration.
   The historical count is 11, not a substitute for an action-time count.
4. Start only approved compute. Require actual current pinned build, matching
   boot, idle preflight, disk below 80%, no swap/OOM. The freshness gate may add
   one read-only readiness control after a long credential dialog; do not hide
   that control in the budget.
5. Review source settings in this window, explicitly approve the one complete
   inventory, and let the user enter the reader credential privately. Immediately
   retain the new operation UUID, configuration hashes, boot, inventory control
   ID, phase and submission time. This is not a second retry of the old job.
6. Obtain live-process proof for this exact new operation, then perform the one
   approved host interruption while that evidence is current. If the operation
   already finished, preserve its report and mark the active-crash gate
   **unexecuted**. Do not replay it, enlarge the dataset, block SQL, pause the
   worker or submit another inventory to make the timing convenient.

## Exact pinned guest process identity

The reviewed historical artifact is Linux AMD64
`2.4.0-dev.d40d6ccc9a4d`, full commit
`d40d6ccc9a4ddf6e2ca626392cd7bf83140ed6c7`, archive 37,197,546 bytes,
SHA-256 `2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`.
These are expected pins; only fresh guest evidence establishes the installed
identity in this session. No new artifact is needed or authorized.

Verified directly against that Git commit:

- Unit: `agefreighter-assessment-<NEW_OPERATION_UUID>.service`.
- Unit's main process:
  `/usr/local/bin/agefreighter-tools runner worker --workflow ae952310-5eba-42b6-9fe3-9db00e93cdac --operation <NEW_OPERATION_UUID>`.
- Inventory child:
  `/usr/local/bin/agefreighter inventory /var/lib/agefreighter/workflows/ae952310-5eba-42b6-9fe3-9db00e93cdac/<NEW_OPERATION_UUID>/job.json --format json`.
- Unit has `Restart=no`, `RuntimeMaxSec=1800`, `MemoryMax=4G`,
  `MemorySwapMax=0`, `KillMode=control-group`, no install-on-boot section.
- `worker.claim` is create-only. A second worker cannot automatically claim the
  same operation. The Windows/macOS editor is not the guest worker's parent.

`internal/runner/manager.go` defines the unit and worker claim;
`internal/runner/linux.go` fixes executable paths;
`internal/runner/protocol.go:Arguments` fixes the inventory arguments.

**Important:** `Manager.Status` reads `state.json` and compares boot identity.
A receipt saying `running` is not independently a live process observation.
Likewise, configured `MemoryMax` is not an observed RSS measurement.

## Minimal sanitized read-only guest observation

Use only the already approved Run Command observation channel on this runner.
Each new managed observation is itself a counted control, not an ARM GET.
Do not change service state. The payload may read these allowlisted fields:

1. Guest UTC time and boot UUID.
2. For the exact operation's unit, `Id`, `LoadState`, `ActiveState`, `SubState`,
   `MainPID`, `ControlPID`, `InvocationID`, `NRestarts`,
   `ExecMainStartTimestamp`, `ExecMainStartTimestampMonotonic`,
   `ActiveEnterTimestampMonotonic`, `MemoryCurrent`, `MemoryPeak`,
   `MemorySwapCurrent`, `MemoryMax`, `MemorySwapMax`, `Result` and `ControlGroup`.
   Use `systemctl show <exact-unit> --property=<explicit-comma-list>`;
   unsupported/missing fields stay unavailable, never silently zero.
3. Require loaded/active/running, positive MainPID, expected
   `/usr/local/bin/agefreighter-tools` executable, and a live direct child with
   executable `/usr/local/bin/agefreighter`. Read only numeric PID/PPID/start
   identity, executable link and RSS for those exact PIDs. Do **not** emit
   environment, arbitrary command lines, `job.json`, `secrets.json`, source CA,
   raw stderr, journal text or database contents.
4. Root filesystem usage, numeric swap-used bytes, and only bounded numeric
   OOM event counts. The existing guest health implementation demonstrates
   fail-closed disk/swap/kernel-OOM checks (`internal/runner/health_linux.go`):
   an unreadable journal or unsupported counter is unknown, not zero. During
   this operation `idle:false` is expected; never manufacture idle health.
5. Report both main-process and inventory-child RSS, separately from aggregate
   cgroup `MemoryCurrent`. If supported, retain peak/cgroup OOM counters as
   separate metrics; do not claim they are process RSS. Apply the approved
   4 GiB/no-swap/no-OOM safety ceiling and stop on uncertainty/violation.

Capture the service InvocationID and each PID's start identity before and after
the local crash. The strongest proof is the **same invocation and child still
running after independent confirmation of the host exit**. A completed report
can instead establish continuity if its same-operation start/finish evidence
unambiguously brackets the host exit; cross-machine clock uncertainty must be
accounted for. A sample that was active only before the exit is insufficient
when it might have finished before the crash. If uncertain, record inconclusive;
do not obtain a second crash or inventory without new authorization.

## Binding the normal window to one Extension Host PID

Root can combine these independent local sources; none alone is a PID guess:

- In the intended normal window, **Developer: Open Process Explorer** identifies
  VS Code process roles and PIDs. Confirm the normal window/profile, not one of
  the disposable fixture development hosts. An extension-host label alone is
  insufficient when several windows/hosts exist.
- That window's **Log (Window)** / renderer log startup entry is
  `Started local extension host with pid <PID>.` The paired window's Extension
  Host log starts `Extension host with pid <PID> started`. The window's log
  location and UI title bind this evidence to the intended window/session.
- The normal main-process log records
  `Extension host with pid <PID> exited with code: <code>, signal: <signal>.`
  Retain only those exact nonsecret startup/exit lines, not whole logs. These
  strings were verified in the local installed VS Code application code, not
  by reading user logs or inspecting current processes for this checklist.
- A side-effect-free process metadata check of that exact PID must agree with
  the app's Helper (Plugin)/Extension Host role, parent application, and recorded
  start time immediately before interruption. A current lock's owner PID/boot,
  when present, is useful additional agreement—not a substitute for window
  binding and never a reason to create a lock.

`code --status` from the exact normal application/profile may corroborate role
and process IDs but is not sufficient by itself to disambiguate same-binary
windows. Do not rely on log files from an earlier app session or PID reuse.
No kill command, all-process match, wildcard target or PID-discovery automation
is supplied here. Root performs the separately approved single exact action.

## History preservation and no replay

At installed implementation `3bc0069`, `assessmentActive` is false only for a
finished assessment with a sealed report hash (and no active catalog).
`startAssessment` then appends that exact previous assessment to
`assessmentHistory`, enforces at most 16 history entries, creates one new UUID,
and persists its intent before dispatch. Source mappings must remain reviewed;
an existing migration freezes them. This path is supported, not a record edit.
The same commit's `runnerAssessment.test.ts` explicitly covers a finished sealed
assessment followed by a new inventory and preserved historical operation.

Preserve old accepted operation `3a4ffae1-8727-464b-8968-40bed2f42c69`, its exact
assessment metadata, original imported report bytes/hash and transfer evidence.
Recheck the actual history count before starting; historical documentation is
not a current store read. Capture only allowed nonsecret record fields.

After the host exit, reconnect to this workflow and restore source assessment;
do not approve inventory again. Pending inventory/status controls reconcile by
GET; subsequent status controls use the **same assessment operation**, a new
control resource ID and action `status`. Retain an allowlisted timeline of
`guestCommand.id/operation/action/phase/submittedAt` to distinguish those reads
from a second inventory request. Never decode protected parameters for proof.

An active cloud operation need not hold a local `.lock`: the watcher releases
it between steps. If a dead-owner lock really remains, use the normal native
reviewed lock-recovery command and preserve its archive. If no lock remains,
record that fact. Do not create or erase a lock to fit a test expectation.

## Cumulative managed-control budget

This B10-only plan assumes **no B09 receipt DELETE or extra readiness pair**.

| Counted addition | New controls |
| --- | ---: |
| Historical baseline, must re-observe | 11 already retained |
| Initial current readiness | 1 |
| Automatic freshness renewal after prolonged input, if needed | 0–1 |
| One new inventory dispatch | 1 |
| Independent before/after guest observations, if separate controls | 2 |
| Report export for the one new report | 1 |
| Subtotal before new status controls | 16–17 total |

Plan to stay at **24 or fewer**, reserving one unused slot below the hard 25
limit. Thus only 8 or 7 additional status controls fit that plan. Actual existing
controls, rejected submissions or other approved activity can reduce it.
GET/list reconciliation does not create a control. Report Blob GET/import does
not create another control beyond the separately approved export.

The watcher runs up to 15 steps, 120 seconds apart, for at most 30 minutes per
invocation. Starting with a pending inventory, steps alternate reconciliation
GET and a new status PUT, so one full invocation can create seven status
controls. Reopening starts a new watcher bound: count **both sides of the crash
cumulatively**, not seven per session. Cancel monitoring when necessary to
preserve budget; this does not cancel the guest inventory. No automatic retries,
unrelated receipt deletion or capacity bypass is allowed.

Finish by importing only the sealed new report, independently checking old
evidence unchanged, runner deallocated and source stopped within the approved
window. A successful inventory is source evidence, not migration or P1 target
qualification. Keep this signed-in active-crash result separate from earlier
unit, isolated-host and native synthetic-cancellation evidence.
