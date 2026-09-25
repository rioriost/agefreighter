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
- Follow-up offline setup utility: `npm run typecheck` and
  `npx --no-install tsx --test src/test/unit/p1DownloadCase.test.ts`: **59 pass,
  0 fail, 0 skip**. Tests use only synthetic snapshots/receipts and disposable
  temporary directories. They cover both original qualification phases, both
  negative expectations, maximum report length, guest-receipt binding, identity
  and profile refusal, nonsecret native approval/Cosmos references, credential
  and ambiguous JSON rejection, input size/UTF-8, operator-path refusal,
  symlink/hardlink refusal, existing directories, concurrent setup, and actual
  offline CLI success/sanitized argument failure.

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

Prepare cases outside the native import operation with the test-only
`extensions/vscode/src/test/helpers/prepareP1DownloadCase.ts` utility. It uses
production `RunnerStore.write` in a newly created private staging store, then
publishes the one disposable record with a create-only hard link. This prevents
the store's normal replacement semantics from overwriting a competing file.
It refuses known operator storage paths, symbolic links in any input ancestor,
hardlinked inputs, and any pre-existing case root, including an empty directory.
It accepts only explicit, owned, bounded nonsecret record/guest-receipt files;
it never discovers operator storage or reads credentials. All input validation
finishes before output is created. Keep
the original workflow/operation identifiers so the genuine Azure storage
ownership and exact report-capability checks remain in force. These are
disposable local test records, not newly qualified migrations.

Run this from `extensions/vscode` only after the record and original guest
export receipt have been explicitly supplied as reviewed nonsecret exports:

```sh
npx --no-install tsx src/test/helpers/prepareP1DownloadCase.ts \
  --record /absolute/approved-exports/record.json \
  --manifest /absolute/approved-exports/guest-export-receipt.json \
  --case-root /private/tmp/af-b12-0123456789ab \
  --scenario wrong-sha256 \
  --acknowledge-nonsecret-inputs
```

Use a new 12-lowercase-hex suffix for every case. On macOS the destination must
be directly under canonical `/private/tmp`; `/tmp` is a symlink alias and is
refused with guidance to use `/private/tmp`. Other POSIX hosts use the canonical
OS temporary directory; Windows is not supported by this setup utility.
For the second case use another new root and `--scenario wrong-length`.
The guest receipt must have exactly `workflow`, `operation`, `jobId`, `sha256`,
`bytes`, and `exported: true`, all bound to the original record. A bare hash/size
manifest or a verification receipt without export confirmation is insufficient.
An exported or previously passing P1 phase is admitted; a submitted, verified,
exporting, failed, or unknown phase is refused.

The utility preserves original raw bytes under `originals/` with read-only file
permissions, and records their hashes, original guest seal, deliberately wrong
expectation, precise field changes and the disposable record hash in
`evidence/case.json`. This is logical read-only retention, not tamperproof
storage. `user-data/` and `extensions/` are separately created under the case
root; the seeded store contains exactly one record and no reports. Successful
output prints only those local paths. Check success and `evidence/case.json`
before using a prepared case. If setup fails, any partial directory remains
for inspection and cannot be reused. No automatic deletion or repair occurs.

Known credential/token fields, signed URLs, private keys and credential-bearing
connection strings are refused recursively. Unknown top-level record fields and
unknown qualification/receipt fields are refused. Only narrowly checked native
cost/resize audit structures and fixed nonsecret ARM/environment/default-Azure
references are admitted. These checks are conservative and cannot prove that
arbitrary opaque strings contain no secret; the reviewed-export requirement and
explicit acknowledgment remain necessary. The files' claimed guest provenance
and Azure authorization are not authenticated by this offline utility.
The helper is under `src/test`, excluded by the existing VSIX `src/**` rule,
and has no production registration, network call, native launcher or adapter.

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
