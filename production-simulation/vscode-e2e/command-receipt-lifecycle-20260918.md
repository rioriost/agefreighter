# Managed command evidence lifecycle: local stages 1–2

Status: local implementation/regression only; not installed-GUI or Azure qualification.

## Problem and scope

The Gremlin GUI qualification needed explicitly approved, archive-first removal
of three old readiness controls to make room for further verification controls.
The extension's 25-record admission threshold remains unchanged. That observation
does not establish an Azure service quota. Manual cleanup is still a release gap.

Stage 1 (commit `6ce5434`) retains successful readiness evidence before the next guest command can
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

## Stage 2: reviewed single-record removal and GET-only recovery

Implemented locally after stage 1; no live cloud call, VM start, extension
installation or removal has been performed for this change.

- A separate native Command Palette action reviews exactly one sealed old
  readiness record. Latest readiness remains protected even after a status
  control has replaced the current command pointer. Other current/history
  operation references are preserved, not just the current command ID.
- Admission requires a matching owned, **already-deallocated** VM, matching
  instance identity/placement, no active or uncertain workflow operation, and
  fresh ARM `Succeeded` provisioning/execution with exit zero. Exact dispatch
  script, bounded expected readiness output, timestamps and receipt must agree.
  No legacy auto-adoption and no cleanup-triggered VM start/deallocation exist.
- A second allowlisted archive binds the exact receipt and live observation.
  Archive bytes/hash/read-back and directory synchronization precede a durable
  single-use intent. Account, trust and resource evidence are rechecked after
  approval and archiving. The approved account also binds the narrow transport.
- DELETE success/acceptance is not completion. A subsequent explicit action
  performs GET only and accepts absence only with the retained matching archive.
  Lost acknowledgements, service refusals and crashes do not replay DELETE.
  Reappearance invalidates a previous absence observation. Arbitrary service
  errors/readiness output are not copied into user-facing failures or the ledger.
- API shape/200/202/204 responses checked against the official
  [2024-07-01 Compute specification](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/compute/resource-manager/Microsoft.Compute/Compute/stable/2024-07-01/runCommand.json).
  The implementation does not invent an unsupported conditional-delete guarantee.

### Stage 2 limits and review

1. Pending/Updating remains a **blocker**, not a successful historical receipt.
   The manually archived September 18 controls included `Updating` even with
   successful execution; this candidate intentionally will not remove those.
2. A durable intent that was never dispatched remains read-only/review-required,
   not automatically reset. New explicit-retry policy is outside this stage.
3. The workflow lock coordinates extension operations, not other Azure clients.
   Native approval requires operator coordination of exclusive control. Fresh
   reads cannot atomically exclude an external mutation between GET and DELETE.
4. Directory fsync must succeed. macOS local storage was tested; Windows and
   remote extension-host filesystem support are not qualified. A crash lock
   still requires operator review, not automatic removal.
5. B08/B09 live lifecycle acceptance and B10 actual forced Extension Host crash
   remain open. Mocked crashes and reopening a real local store are explicitly
   different evidence from a signed-in host crash or an Azure deletion trial.

### Stage 2 local validation

- `npm run check`: TypeScript, **380/380 unit tests**, bundle build pass.
- Tests exercise cancellation, references, ownership, running VM, Pending,
  Updating, malformed/changed output, account/trust changes, expiry, archive
  failure/corruption, persistence failure, and changes during archiving.
- Lost DELETE replies and simulated crashes after intent preserve no-replay
  semantics. A new `RunnerStore` instance reopens real fsynced private evidence
  and reconciles the unknown intent by GET only. Missing archive blocks even a
  would-be 404; a reappearing resource invalidates previously observed absence.
- Native handler adapters verify lock/approval boundaries and fsync ordering;
  the Azure transport adapter rejects other scopes/accounts and missing trust.
- `git diff --check` passed. These are local/inert-adapter tests, not a live
  Windows/macOS signed-in GUI or Azure service result.

### Pinned candidate (before installation)

- Source commit: `dbf017d0dd80835360cc4d6f032708b7ed3aca5c`.
- VSIX: `production-simulation/work/vscode-receipt-removal.X1nhbl/agefreighter-dbf017d-readiness-removal.vsix`.
- VSIX SHA-256: `a1139aad6e6eec7f55312fc0394e6697701422cbaf96e382809880e942b66866`.
- Bundled JavaScript SHA-256: `965a4c4422ff2816668611d937123f9f05377b59bb2cd848f72a29222e74f6a3`;
  exact match between the compiled local file and the extracted VSIX member.
- Packaging reran all 380 tests and build successfully. Package manifest is
  version 2.4.0 and contains both archive and reviewed-removal native commands.
- No VSIX installation, Marketplace publication or Azure operation performed.
  Installation/live qualification must preserve current accepted workflows and
  use separate action-time approval for the exact removal target, if eligible.

### Approved installation and native empty-state check — September 18

The user explicitly approved installing the above exact unpublished candidate
and reloading VS Code, limited to GUI checks without Azure mutations.

- Actual host: VS Code **1.138.0**, commit
  `7debcd0e2acdea1c52de81bf9ee1620444407dda`, Apple Silicon. The earlier
  1.136.1 host version is not the version observed for this checkpoint.
- Before installation, private backups retained the existing extension and
  runner store under `production-simulation/work/vscode-receipt-backup.SknrNo`.
  Extension archive SHA-256:
  `bb7e51e165c4516c636d99f361b92c6f4a45cb07b2431bcb9165a8e52608aea2`;
  runner-store archive SHA-256:
  `89ba27ba15c87c48a876bb8a24a80d283603873a3c7e7f069689ec7f2fb70ac0`.
  These local private archives are not committed.
- Native **Extensions: Install from VSIX** reported completed installation;
  **Developer: Reload Window** completed. Installed bundle SHA-256 matches
  the pinned candidate: `965a4c4422ff2816668611d937123f9f05377b59bb2cd848f72a29222e74f6a3`.
- Both new Command Palette actions are present and activate. Archive reports
  “No sealed readiness receipts are available. Legacy ARM commands are not
  automatically adopted or removed.” Reviewed removal displays an empty
  workflow selector (0 results); it was cancelled without selecting a target.
  An explanatory empty-state message would improve this selector's UX.
- Before/after store inventory is identical: **70 files, 19 workflows, zero
  sealed-receipt workflows, no lock files**. SHA-256 of the sorted filename/file
  SHA-256 pairs remains
  `fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.
  The Gremlin record retains migration `finished` / qualification `pass` and
  operation `b0530700-ccd4-4f33-84fa-0854c8f4037b` unchanged.
- The existing P1 PASS panel was visible before reload. Reload closed its
  non-restored webview; the accepted report/state remain unchanged on disk.
  Reopening via the qualification handler was not attempted: even its retained
  PASS path performs a storage-control check and persists the record. This
  checkpoint does not claim a newly rendered PASS panel after reload.
- Read-only ARM checks reconfirmed the exact Gremlin VM `deallocated` and target
  `Stopped`. No Azure start, stop, deletion, RBAC, credential, network or storage
  policy change was made. No migration or verification worker was submitted.

Installed activation, legacy non-adoption and evidence preservation are proven.
Successful eligible-receipt archiving/removal in the installed GUI, live Azure
removal/GET-only recovery and actual forced Extension Host crash are **not**
proven. Existing legacy workflows do not provide a sealed eligible test receipt;
do not fabricate/adopt one or start cloud resources just to close this gap.
