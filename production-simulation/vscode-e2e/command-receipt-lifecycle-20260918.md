# Managed command evidence lifecycle: local stage 1

Status: local implementation/regression only; not installed-GUI or Azure qualification.

## Problem and scope

The Gremlin GUI qualification needed explicitly approved, archive-first removal
of three old readiness controls to make room for further verification controls.
The extension's 25-record admission threshold remains unchanged. That observation
does not establish an Azure service quota. Manual cleanup is still a release gap.

Stage 1 retains successful readiness evidence before the next guest command can
replace the current pointer. It does not add DELETE support or remove any ARM
record. Existing VM, target, source, RBAC, policies, disks and guest evidence are
untouched. The installed extension remains the separately qualified candidate.

## Implemented and tested locally

- Successful `reconcileGuest` creates an allowlisted, SHA-256-sealed readiness
  receipt bound to workflow, VM, exact command, operation, submitted time, boot
  and installed artifact. Repeated reconciliation is idempotent; conflicting
  successful evidence is not silently rewritten or made fresh.
- Failed/pending/unknown and non-readiness controls do not create such receipts.
  No protected parameters, raw ARM output or source settings are archived.
- A native Command Palette action selects one workflow/receipt and retains a
  create-only local archive with verified bytes/hash, then reads it back before
  displaying success. It requires no Azure adapter, target, running VM or CLI.
- Cancellation, trust loss, changed VM/subscription/receipt and local write
  failure do not display successful export. Referenced receipts are marked for
  preservation; the archive never grants removal authority.
- Historical artifacts remain verifiable after a runner upgrade; legacy ARM
  records are not auto-adopted merely because their name starts with `af-`.

## Still required (B08/B09 lifecycle gate remains open)

1. Review selective removal admission: positive ownership, exact archived
   evidence, no current/history references, no active/uncertain worker, fresh
   resource/account/trust checks and narrowly scoped native approval.
2. Before any DELETE, verify a durable archive and persist an exact removal
   intent. Resolve lost acknowledgements using read-only reconciliation, not
   replay. Cover archive failure, concurrent changes, service rejection and
   extension-host crash. ARM metadata deletion is not guest evidence deletion,
   but must still be explicitly disclosed.
3. Resolve the observed stopped/startup VM `Pending` instance-view behavior
   without assuming historical success is fresh terminal evidence, restarting
   stopped compute solely to inspect it, or weakening the preservation gate.
4. Review/package a pinned candidate; separately approve/install and exercise
   the native action in VS Code. Only then perform a bounded live removal trial
   with fresh time/cost/governance checks. Local tests do not close that gate.

The full readiness control lifecycle is **not complete**. Other branch tests,
forced Extension Host crash, installed-candidate regression and release remain
tracked in [remaining-validation.md](remaining-validation.md).

## Validation and review

- `npm run check`: TypeScript checking, **338/338 unit tests**, and extension
  bundle build passed. Production reconciliation and native panel handlers are
  exercised with inert adapters; no cloud access was supplied to the new panel.
- New storage tests exercise real private local files, create-only publication,
  hash read-back, repeated export, rejected replacement and state replacement.
- `git diff --check`: passed. No cloud execution, VSIX installation, native
  Extension Host crash or live deletion was performed in this stage.
- Review boundary: these are normalized validated readiness receipts, **not full
  raw ARM archives**. They contain no proof of current ARM resource identity,
  absence of external governance changes, or safe deletion. Future removal must
  independently gather and bind that evidence; a matching SHA alone is not an
  admission decision. Existing archives and current controls remain untouched.
