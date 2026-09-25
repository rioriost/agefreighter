# B09 running-VM readiness control removal admission

Status: local implementation and inert-adapter validation only. Native live
removal and lost-acknowledgement recovery remain unqualified; B09 remains partial.

## Observed contradiction and scope

The original native removal gate admitted only an already-deallocated VM.
The [September23 PostgreSQL record](other-cloud-pg-live-20260922.md) separately
records Azure rejecting a managed control DELETE while deallocated, and older
readiness controls reporting Pending with no output or execution timestamps.
The five explicitly approved manual removals did not qualify the native path.
The cause of the missing historical instance-view evidence remains unproven.

This change admits an already-running owned VM with a separate current readiness
control. It does not permit Pending controls to be deleted or converted into
successful receipts, and does not start compute, dispatch readiness, raise the
25-control limit, adopt legacy records, or extend any approved runtime window.

## Native admission and preservation

- Select exactly one sealed, unreferenced successful readiness receipt. Current,
  latest and operation-history references stay protected. Local active/uncertain
  work, deployment and previous unconfirmed removal intents block admission.
- Require the exact owned VM, unchanged placement/configuration/instance identity,
  Succeeded provisioning and Running power state. No VM start/stop is performed.
- Require a separate newer same-boot readiness receipt matching the current
  guest-command pointer, guest readiness and pinned artifact. Its original
  observation must be at most five minutes old, with idle worker, disk below
  80%, no swap and no OOM events. Reading old ARM output does not renew freshness.
- GET both exact controls and require Succeeded provisioning/execution, exit zero,
  bounded matching output, timestamps, constant script and permitted command
  settings. Pending, Updating, absent output or changed evidence blocks removal
  for either control. A local seal alone is never current health evidence.
- Bind the selected exact historical receipt, separate current receipt, both
  validated live observations, VM configuration and account into a v2 normalized
  archive. No protected parameters or raw service errors are retained.
- Obtain one native approval, recheck account/trust/workflow and live evidence,
  durably archive/read back, recheck again, then persist a single-use intent
  before one DELETE. Approval expires no later than the current readiness's
  original five-minute validity boundary. Latest readiness and guest evidence
  remain unchanged. Removing the ARM control is irreversible through this action.
- Lost replies, service refusals and retained intents reconcile by GET only;
  HTTP acceptance is not completion. Exact 404 with verified bound archive is
  required. Existing v1 intents retain GET-only recovery even with compute off;
  v1 previews cannot authorize new dispatches.

The workflow lock coordinates this extension. Azure does not provide an atomic
GET-and-DELETE guarantee here; the native approval requires exclusive operator
coordination with other clients. Recent health proof is bounded evidence, not a
continuous observation. Host-crash recovery remains a separate B10 boundary.

## Local verification

- TypeScript checking, full unit suite and bundle build pass (`npm run check`):
  **542/542 tests** at the shared-worktree checkpoint, no failures or skips.
- Focused `runnerReceiptRemoval` and `runnerReceiptRemovalPanel` suites:
  **77/77 pass**. `git diff --check` also passes.
- Focused controller/panel coverage includes successful archive/intent ordering,
  cancellation, stopped VM, stale/future/busy/unsealed readiness, artifact/boot
  mismatch, Pending/missing/changed current proof, historical Pending refusal,
  latest-reference protection, active history/deployment and uncertain removal.
- Candidate/current control changes, current receipt changes and VM stop during
  approval or archive block deletion. Existing lost-reply/crash/archive-corruption
  checks remain. Added v1 GET-only recovery and stale v1 preview refusal checks.
- Native handler tests verify disclosure, empty state, cancellation, trust/account
  loss, archive/intent synchronization and exact one-command transport.

All ARM/UI adapters were inert. No cloud call, credential access, installation,
operator-store edit, guest execution or deletion was performed by this change.

## Remaining live boundary

After candidate review/installation and a separately bounded compute approval,
an operator may use a dedicated same-boot pair of naturally successful readiness
controls: an older unreferenced candidate and a fresh current control. Both must
still satisfy the full live admission. Observe actual modal cancellation first,
then separately approve one exact native removal, verify archive-before-intent,
and use native GET-only absence reconciliation. Preserve all current job,
migration, canonical result and readiness controls. Do not assume the remaining
controls on any existing VM are removable or create capacity by weakening gates.

Historical Pending remediation is still outside this path. A live service refusal,
lost reply or host crash must preserve the archive/intent and stop automatic
mutation; it cannot be resolved by resubmitting DELETE under the same intent.
