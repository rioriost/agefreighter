# B12 retained canonical evidence: real Extension Host rejection checks

September22,2026,02:15UTC. Scope: real VS Code1.138.0 Extension Host with an
isolated disposable profile; synthetic report envelopes and inert ARM replies.
This adds host-level evidence, not signed-in Azure transfer-fault qualification.

## Implementation and cases

`extensions/vscode/src/test/suite/p1Retained.test.ts` invokes the production
`qualifyP1` controller, canonical parser, real on-disk `RunnerStore`, locking,
sealed-byte reader and actual VS Code webview/tab API. The VS Code API is not
mocked. Only the storage-ownership ARM adapter is inert and GET-only. Every
credential/capability/upload adapter throws if accessed. Reports have synthetic
job/target envelopes around the tracked frozen canonical leaves, not new target
reads. The tested path is reopening retained evidence, not fresh Blob download.

| Case | Expected / observed |
|---|---|
| Complete retained canonical report | One actual `Verified P1 migration` tab; one persistence |
| Foreign envelope job / foreign target job | Rejected; no PASS tab or persistence |
| Missing / duplicated / reordered range | Rejected; no PASS tab or persistence |
| Forged matching leaf hashes under the accepted root | Recomputed root rejected; no PASS tab or persistence |
| Wrong record count / failed comparison / non-read-only receipt | Rejected; no PASS tab or persistence |
| Truncated retained JSON / changed retained bytes | Rejected; no redownload, PASS tab or persistence |

All11negative cases preserve both pre-attempt record and report bytes. Each case
uses a separate disposable store; negative reports are never injected into the
operator's accepted workflow. For reopen tests the seeded local phase is `pass`:
the assertion is no new PASS presentation/persistence, not automatic rewriting
of retained historical state. Malformed JSON is deliberately seeded only in the
test store, since normal report publication already rejects it.

## Execution and evidence

- `typecheck`, host compilation and extension build: PASS.
- Full unit suite: **457pass,0fail,0skip**.
- Real host suite: **25pass**, including the12new cases above and13existing
  counts/source/editor activation cases; process exit0.
- Host executable resolved from installed bundle: `Contents/MacOS/Code`.
  First launch using the historical `Electron` name failed ENOENT before any
  host tests ran; corrected launch succeeded. This is not a test pass/skip.
- Isolated profile/evidence retained at `/tmp/af-extension-host-rbP6Cl`.
  `p1-retained-results.json` SHA256:
  `9dbff26c5d050677b254d9972c68ce0541827ab8e2e9a1c654177821fad4dd68`.
- Real tab inventory asserted; no pixel/VoiceOver verification claimed.
- Cloud adapters made no real requests. Built-in VS Code background traffic
  is outside that assertion; this was not a network-capture test.

The test launcher supplies an explicit disposable-profile marker, checks the
development extension path, uses a short macOS temporary path to avoid IPC path
limits, and restores the profile-local development opt-in afterward. The
profile's final settings file is empty. No signed-in operator profile/secret
store was used and no new artifact was installed into normal VS Code.

All82operator files are byte-identical before/after, aggregate name/content SHA:
`210699618b47cae309efc601398bfb171ed37248675d1436af2363c04735eb4a`.
Rebuilt production bundle still matches the installed a4b61d8 bundle SHA:
`8f0061a16bf2d42abe1a41cd6f805697316f70286a905c20e95d56b80fdd2a8e`.
Only test harness/source and documentation changed. No Azure startup, migration,
source mutation, credential/RBAC/network change or deletion occurred.

B12 remains **partial**: signed-in invalid-import/Blob-transfer fault evidence
is still missing. Existing20production-controller adapter tests and this real
isolated-host batch must not be described as an actual Azure outage. Overall
defined branches remain5pass/7partial; base routes remain9/9qualified.
