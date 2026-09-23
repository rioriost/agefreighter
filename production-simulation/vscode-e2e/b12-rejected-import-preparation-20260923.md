# B12 rejected import classification and fresh-download preparation

September 23, 2026. B12 remains **partial**. This change supplies local
classification and regression coverage; it does not establish a signed-in Azure
transfer fault or a new P1 migration qualification.

## Behavior and evidence boundaries

The existing controller already refused invalid canonical reports before
persisting a P1 `pass` or opening the `Verified P1 migration` tab. Hash-valid
JSON objects were intentionally retained before canonical validation. This
was not a demonstrated false-PASS defect, and those immutable original bytes
remain retained for investigation.

Canonical rejections now have closed categories: `json-shape`, `profile`,
`identity-outcome`, `coverage`, `range`, `canonical-root`, and `leaf`. After
successful byte retention, `P1RejectedImportError` carries an immutable
`status: rejected`, `retained: true` envelope containing the guest operation,
SHA-256 and byte length plus the reviewed job and profile. Its UI text explicitly
says the evidence is retained and is not an accepted P1 result. This envelope
is returned with the error; it is not a new persisted sidecar or workflow phase.
The underlying report file remains the original bytes. The controller does not
rewrite a historical `pass` record when reopening rejects its evidence, and
does not display a new PASS tab in that case.

Transport rejections distinguish HTTP status, missing body, unexpected encoding,
length, SHA-256, UTF-8/JSON, and transport interruption. They expose fixed text
and a category only, never a capability URL, response body, or raw error cause.
Invalid manifests and capabilities still fail before a GET, and are not
classified as observed Azure failures. An import uses one bounded GET with
redirect refusal and no retry. Failing stream cleanup cannot replace the first
sanitized rejection. Malformed JSON/non-object JSON and invalid UTF-8 fail byte
admission before retention, consistent with prior behavior.

## Local validation

- `npm run typecheck`: PASS.
- `npm run test:host:compile`: PASS; compilation is not host execution.
- Focused unit invocation: `npx tsx --test src/test/unit/p1Qualification.test.ts
  src/test/unit/p1QualificationPanel.test.ts src/test/unit/runnerReport.test.ts
  src/test/unit/runnerBlobRejection.test.ts`: **65 pass, 0 fail, 0 skip**.
- Controller tests exercise the production parser/download logic with inert
  VS Code, store, ARM and HTTP adapters. Rejections preserve exact invalid
  canonical source text, leave workflow state unchanged, and create no PASS
  panel. They also assert the guest seal and reviewed identity in the typed
  rejection. These are not operator-profile or signed-in results.
- The existing 12-case `p1Retained.test.ts` real-host suite now also checks the
  typed rejected-evidence envelope. This worker compiled it but did not launch
  it; the parent coordinator runs host suites serially and records outcomes.
- No live Azure request, operator-store access, GUI action, installation,
  credential access, or accepted workflow mutation was performed by this worker.

## Concrete later signed-in native procedure

The supported production entrypoint already exists:
`agefreighter.continueRunnerExecution` (Command Palette title
`AGEFreighter: Continue / Verify Linux Migration`) → select the retained target →
`Qualify / reconcile full P1 digest (development only)`.
`registerRunnerMigration` builds the real `AzureSession` and `RunnerStore`
under that window's `context.globalStorageUri/runner-v2`; the action calls the
production `qualifyP1`, `verifyTransferStorage`, `downloadReport`, and `verifyP1`.
There is no fault-injection switch or replacement network adapter in this change.

Use an explicitly approved, separately signed-in disposable VS Code user-data
profile and extension directory. Installation/sign-in and access to an exact
existing report must be approved before the later session. Keep the production
operator profile closed or outside the test window. Authenticate interactively;
never copy SecretStorage or place SAS/token values in test files or evidence.
Do not reuse the automatic isolated-host launcher for this run: its tests are
deliberately unsigned-in with inert Azure adapters.

Prepare cases outside the native import operation with a reviewed test-only
setup utility using `RunnerStore.write` in the verified disposable profile.
That utility is not implemented or registered by this change. It must refuse
the normal operator storage path, an existing record/report, symlinks, and a
nonempty case directory. It takes only an explicitly authorized nonsecret
record snapshot and independent guest manifest; it must not discover or read
the operator store itself. Retain that original snapshot and manifest unchanged
and hashed in a separate evidence directory before creating any case. Keep
the original workflow/operation identifiers so the genuine Azure storage
ownership and exact report-capability checks remain in force. These are
disposable local test records, not newly qualified migrations.

For each case, use a separate disposable profile/store with exactly one copied
record. Require a finished migration with passing counts, the correct original
qualification job/profile/command identity, and an already exported authorized
report. Set only the disposable qualification phase to `exported`, so the
native import takes the fresh-download branch even if the original retained
record had already reached `pass`. Do not copy any retained report file into
the case store. Retain a before/after diff that explicitly identifies this local
test setup and distinguishes the original guest seal from the test expectation.

| Case | Disposable test expectation | Required native observation |
|---|---|---|
| Wrong SHA-256 | Preserve operation/bytes; replace only the expected digest with a valid, different 64-hex digest | Actual authorized Blob GET, complete body checked, SHA-256 rejection; no retained report, PASS tab, or state promotion |
| Wrong length | Preserve operation/digest; use a different valid bounded byte length | Actual Blob response, length rejection; no retained report, PASS tab, or state promotion |

The wrong-length case may reject the real response's `Content-Length` before
reading the body. Record that distinction; it is not proof of a truncated
Azure body. These two cases prove signed-in rejection of deliberately wrong
local expectations against a real fresh download. They do not prove that Azure
corrupted, truncated, redirected or denied a request. No altered Blob is needed.
Creating separate malformed or canonical-invalid Blobs requires a later,
separate approval for the exact disposable blob/container scope and immutable
create-only upload; it is outside this preparation.

Invoke the normal command and select only the P1 reconcile action in the real
native picker. Do not invoke the controller directly and call that native GUI
evidence. No accepted record is edited and no production behavior is replaced.
With an `exported` record, this route performs storage ownership reads and
just-in-time read capability issuance, then the bounded GET; it does not run
verification, export, start compute, or touch graph data. Azure authentication
and user-delegation capability issuance are real service interactions and must
be included in the session's approved scope.

For each run, retain candidate/version/hash, exact case setup hash, original
manifest hash, disposable profile path, account/subscription identifiers,
timestamped native picker/error captures, category, no-new-PASS tab inventory,
and byte-identical before/after local record evidence. Record the real request
through already authorized sanitized service telemetry where available; never
enable diagnostic settings or log full request URLs as an implicit part of
this test. If request evidence is unavailable, report that coverage limit.
Any unexpected successful import or mutation ends that case for investigation;
do not rewrite its evidence or automatically replay. Preserve disposable
evidence after the run. B12 stays partial until actual results are reviewed.
