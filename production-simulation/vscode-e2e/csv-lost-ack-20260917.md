# B08 committed CSV acknowledgement-loss qualification

Status: **live Azure production-transport and installed-GUI fault trials PASS**.
B08's defined cases are complete. This is not overall release qualification.

## Installed-GUI result — September 17, 05:10–05:14 UTC

The user approved the isolated unpublished extension at action time. The
temporary VSIX was installed, its bundle hash checked, and the window reloaded.
The signed-in GUI reconnected to the existing isolated workflow without replay.
The native picker added only `Ack-Supplier.csv`, assigning file UUID
`672664ee-8b22-47cd-8bcd-b30966f47135`. Earlier selections were preserved.

- 05:10:44.118 UTC: Azure returned BlockList **201**, ETag
  `0x8DF147A06B67A0E`. The adapter recorded that success and dropped the response.
  GUI displayed acknowledgement-uncertain; independently read saved state was
  `prepared`. No automatic retry followed. This is an injected client response
  loss after real Azure success, not an actual Azure service outage.
- 05:11:13 UTC: independent GET matched all 8,797,607 bytes and the expected
  SHA-256. The native upload confirmation was explicitly accepted again.
- 05:11:25.954 UTC: the retry issued **one HEAD 200, zero PUTs**, kept the
  same ETag and changed the saved phase to `uploaded`; the GUI error cleared.
  This is not `verified` (Linux import/sealing was intentionally not performed).
- The regular candidate VSIX was restored successfully, its exact installed
  bundle hash verified and fault marker confirmed absent. After Reload Window
  and reconnection, the normal GUI visibly retained `uploaded`, the earlier
  negative `failed`, and positive `verified` states. No assessment had started.
- 05:13:38 UTC: independent full GET again matched the expected bytes/hash and
  ETag. All four pre-existing Blob ETags were unchanged. All 64 other saved
  workflow/report JSON artifacts retained aggregate SHA-256
  `917a29e7457ef2b6f235970141cdc4c262c10543329aa04d189e7c18ee4fc8f8`.
- Fresh Azure inventory showed **9 VMs deallocated and 17 Flexible Servers
  stopped**. No compute start, role/network change, Linux import, assessment,
  target creation, migration or deletion occurred. Storage retention continues.

Structured result: [GUI evidence](evidence/csv-gui-lost-ack-20260917.json).
The temporary VSIX was not published. Its archive SHA-256 is
`402afc8097a0cc04daa8ac018375b40428448cdad6d75c8176c99a2a83c8196a`.
The retained GUI transport trace SHA-256 is
`13de087b222d289de6428fe9b2b1108e738c73a5304834a1a6d8ea3b96e4780e`;
the create-only commit witness SHA-256 is
`b5a40fe8c4beaeebbe08ec02f2f4467423ba272f58a43edc33d9056f3a2bdff4`.

## Scope and review

Use the already authorized isolated account `afbd3b66801e184d788f36f4`, owned by
workflow `bd3b6680-1e18-4d78-8f36-f43467a09a0a`. Create only new test Blob
identities. Do not overwrite the earlier negative/control files, edit saved
GUI state, start VMs/DBs, grant roles, change networking, or delete evidence.
The ceiling remains USD800 and deadline September 20 16:14 JST. All nine
retained VMs were independently observed deallocated before this trial.
The existing account still has the approved exception tag, HTTPS-only/TLS1.2,
public endpoint, shared keys disabled and anonymous access disabled.

The fault must occur **after a genuine Azure block-list commit HTTP 201**, not
before upload or by merely setting a GUI phase to prepared. Drop that response
at the injected transport boundary, retain the commit ETag, and require an
uncertain error without an automatic retry. Explicitly retry later and prove
zero further PUTs, unchanged ETag and full readback/hash equality. This is an
intentional client-side response-loss simulation, not a claimed Azure outage.

## Actual production-transport result

`extensions/vscode/scripts/qualify-csv-lost-ack.ts` imports the actual production
`inspectCSV`/`uploadCSV` implementation. It is excluded from the extension
package. Its two phases require an explicit isolated-write flag, fixed trial
scope/deadline, ownership/security checks, exclusive evidence creation and a
new file UUID absent from GUI state. Tokens are memory-only and never logged.
The original frozen Supplier CSV is read-only, 8,797,607 bytes, SHA-256
`0ecaaca3879b11a4bc76835c23f37cda9bfac7d5ed457ad50937170876d861fe`.

New test file: `ee1d762a-d24a-4df9-bd68-7685219e08ee`.

- 04:00:19 UTC: HEAD 404, two bounded PUT Block requests 201, PUT BlockList
  201. The adapter retained the real success witness before throwing. The
  production uploader returned acknowledgement-uncertain and made no retry.
- 04:00:45 UTC: a separate explicit invocation reconciled with **one HEAD 200,
  zero PUTs**. Independent GET read all bytes and matched the frozen SHA/size.
  ETag remained `0x8DF1470309BB702`.
- Saved GUI workflow bytes are unchanged. No Linux import, source assessment,
  target, migration or VM start occurred.

Retained evidence hashes:

| Artifact | SHA-256 |
|---|---|
| Commit witness | `a51721d98ce35eb609a9e4f46118ff88f0e7f9ae96350bdacc033b1c69abbed0` |
| Fault result | `7c3426a855e5c42d95c35018cb257d2cb17dccb1107775e37c28cb2e535e56b6` |
| Reconciliation result | `bdc336a749f91badf49ce48c8870b251705a8bbdb45da502cb7247fd4ce9ce5d` |

This closes the live transport question, not the installed-GUI/controller
interaction. Existing controller tests remain simulated evidence.

## Installed-GUI trial design and prepared artifacts (pre-execution record)

Use the same isolated workflow/account and select the independent
`Ack-Supplier.csv` copy through the native file picker, creating a fresh file
UUID. Retain all previous selections and operations. Never import this extra
copy or upload the intentionally corrupted negative Blob again.

The temporary package is built by
`extensions/vscode/scripts/build-csv-ack-qualification.mjs` into a separate
ignored trial directory. It changes only the transport argument of the exact
reviewed CSV upload call during that special build. Instrumentation applies
only to this workflow, this exact local file path and the expected size/hash;
its network URL must equal the workflow's derived Blob destination. It records
method/status/ETag, never request headers, tokens or contents. Only a successful
BlockList 201 is dropped, with a create-only commit witness and in-process
one-shot guard. The retry must use the normal GUI confirmation and production
controller; do not patch persisted phases. Deadline applies before requests.

The normal build script is untouched. A fresh normal build was scanned and
contains no `AF_CSV_ACK_QUALIFICATION_ONLY_20260917` marker. Both scripts are
excluded by the ordinary package rules. The special artifact is marked
**qualification only / never publish** in its display name, description and
README. The regular replacement VSIX is prepared for restoration after the
trial; the installed extension has not yet been replaced.

- Temporary bundle SHA-256:
  `005253bf0f614d56d66c2ce7c293c9247c31656d02f8668f6bfb83a37038594c`.
- Restore bundle SHA-256:
  `3061aea43c315113f47df3ad1da31dc68ad73c27b47a18571ddfb66bf742e929`.
- Restore VSIX SHA-256:
  `11e8c511656d502dfb868c56e7691322c0a05c94aaa312436fc4abd4e46a759a`.
- Typecheck, script-specific TypeScript check and 235 unit tests PASS.

Action-time approval was requested before installing/running the unpublished
temporary extension. After approval: install/reload, select the new file,
approve upload, observe uncertain error and retained `prepared` phase, retain
real commit witness, explicitly retry, prove only HEAD for that file and no
changes to earlier Blobs, then restore/reload the normal extension. Preserve
both installed bundle hashes, exact operation evidence and GUI observations.
Do not mark the GUI case PASS or B08 complete before those steps.
