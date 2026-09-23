# B09 deployment consent binding — September 23

Status: **defensive correction and local regression PASS; updated installed-GUI
qualification remains pending**. No deployment, VM start, permission change or
cloud fault was performed to diagnose or validate this correction.

## Correction

Independent review found that the deployment panel validated its in-memory
preview before opening the native approval dialog, then passed the latest saved
record to submission without binding it to the original review. A renewed
preview can retain the same preview hash but have a different expiry/revision.
Validating that later preview does not establish approval of its new contents.

`runnerMigration.ts` now captures the original record's full SHA-256, workflow,
preview hash, expiry and revision before native consent. Under the workflow lock,
the durable reread must still match. The original deadline, panel liveness,
workspace trust and explicit user-level development opt-in are checked again
before intent persistence and before non-GET dispatch, including after awaited
preflight. Submission may advance its own copied durable intent; that is not
mistaken for a concurrent change to the approved record.

If a guard fails after intent persistence but before dispatch, the existing
uncertain-state handling retains evidence without a PUT or automatic retry.
No stale record is rewritten to manufacture a deployable preview.

## Validation

An Astra high implementation agent added22regressions; Astra xhigh independently
reviewed the source/tests and found no blocking issue. Root independently ran:

-45production-panel lifecycle tests plus4new cancellation-fixture contracts:
  **49/49 PASS**.
- Full unit suite: **691/691 PASS**, no skips/cancellations.
- TypeScript check, normal production build and host-test compilation: PASS.
- Diff whitespace validation: PASS.

The lifecycle tests execute the production message handler with inert local
adapters. They assert refusal without dispatch for changed review/expiry,
trust/opt-in loss and panel closure. Their injected timing is not native
multi-window or Azure evidence. The updated product bundle has not been installed
in the user's normal profile in this follow-up. The already approved3bc0069
B12 profile intentionally remains pinned while testing its unchanged import path.

## Finite remaining acceptance

The original plan requires independent generic branches, not every source/SKU
combination. Existing actual quota-denial and missing-artifact refusals are
credited; they do not need another resource creation. Native local lock recovery
already has separate B10 evidence. Remaining B09 work is:

1. The uncredited native approval cancellations in the frozen30-decision ledger.
2. Expired-preview and duplicate-click/multiple-window refusal with no extra
   mutation, using the corrected consent binding.
3. A genuinely uncertain deployment response reconciled by its exact retained
   deployment identity, without replay; synthetic loss is separately labelled.
4. Terminal guest-bootstrap failure remains unready and cannot dispatch a source
   operation; a historical transient bootstrap race is not that proof.
5. Native readiness-removal Cancel, one exact approved archived DELETE, then
   GET404. Optional receipt-specific timing faults do not add more required live
   deletes or replace runner-deployment cases.

See [the bounded runbook](b09-b10-offline-checklist-20260923.md). Existing runner
restart authority does not authorize a new fault-test deployment; exact scope,
budget and stop bounds must be settled before such a mutation.
