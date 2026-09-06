# First Linux runner qualification — 2026-09-05

Status: bounded Linux CSV transport/assessment trial completed on September 6; not a P1 migration qualification. VM independently confirmed deallocated after completion.

September 6, after user reauthentication: diagnosis corrected. Fresh GUI
submission reports **HTTP 400**, and the VM had exactly **25 Managed Run Command
resources**, Azure's documented limit ([official limit](https://learn.microsoft.com/en-us/azure/virtual-machines/run-command-overview)).
The earlier MFA log was real but did not establish the cause of each later
submission failure; GET success and account-list presence were insufficient.
`f9eb593` exposes bounded HTTP status diagnostics and archives absent old
readiness probes without replaying source reads or accepting stale boot proof.
`c9275b2` checks command capacity before persisting/submitting a new request.
127 unit tests and packaging passed.

Archived all 25 command instance views, including checksums and the initial
decoder failure, in [retained receipts](evidence/runner-commands-20260906-01.json)
and pushed them before maintenance. Fresh GETs matched the archived SHA-256 for
all 24 successful terminal commands. Removed only those 24 **ARM command
resources**; kept the failed command, all guest operation directories, CSVs,
seals, disks and storage. Historical command resources are no longer queryable
in ARM, but their results remain in the committed archive. No command was
re-executed during that maintenance. A fresh GUI readiness request at 01:49:30Z
succeeded (exit 0), and the retained Location operation was reconciled without
re-importing it. GUI import continuation succeeded. CI `34004772287` passed.

Independent health at 01:55:54Z: cloud-init done; disk 5% used
(2,973,982,720 / 66,404,147,200 bytes); available memory 7,784,648,704 bytes;
swap and OOM counters zero. Both installed executable hashes still match the
approved artifact below; private runner directory remains root-owned 0700.

## Deployment and installation

- Actual Mac VS Code 1.136.1 GUI, existing Azure Resources login, saved workflow
  `83c6b829-acdc-4405-aa2d-fb2f2d99af9f`. No desktop AGEFreighter invocation is
  used by the guided workflow.
- Dedicated Japan East / zone 1 B2s_v2 (2 vCPU, 8 GiB), Ubuntu 22.04 Gen2,
  64 GiB Standard SSD. ARM submitted 12:48:08Z; GUI reconciled provisioned
  12:49:07Z. No VM public IP or SSH ingress.
- Explicit Standard NAT egress on the dedicated subnet, which retains
  `defaultOutboundAccess=false`. Storage policy exception remains on the transfer
  account only. VM identity Blob Reader is scoped to the workflow container.
- Azure automatic shutdown independently verified `Enabled`, 16:00 UTC, exact
  owned VM target. No automatic start is configured.
- Guest readiness initiated through the GUI at 12:49:29Z and reconciled:
  Linux amd64, `2.4.0-dev.6ff072c0db71`, commit
  `6ff072c0db71a3ae1e42a72e3c336a04ac284df8`, archive SHA-256
  `5fee11ec3fde605ef7e18afaa56131862113348de8ddcb8bc37b52aba1c3620b`.
- Independent guest health at 12:51:30Z: cloud-init done; disk 4% used
  (2,316,439,552 / 66,404,147,200 bytes); memory available 7,734,779,904 bytes;
  swap configured/used zero; `pswpin`, `pswpout`, `oom_kill` all zero.
  Private runner directory is root-owned, mode 0700.
- Installed executable SHA-256:
  - agefreighter: `c99b483b9badbd1a784b93a034991f7c983cd90411b7b21314195ca1253fcd84`
  - agefreighter-tools: `cf8e2f3259619e443217bc9229bd1a8009bbf7084fa1806e7b6b1f005bdf1878`

## Live defects found and corrected

1. Cloud Services pricing shares the ordinary Linux VM SKU and service name.
   The parser now excludes that different product, while retaining ambiguity
   rejection for ordinary Linux meters. `050ee1f`, 117 unit tests and CI
   `33966950563` passed. The real GUI then accepted 0.109 USD/hour.
2. The first `CARRIED_BY.csv` import at 12:50:46Z sent the controller's `phase`
   field as part of a structurally compatible manifest. Go correctly rejected
   unknown fields (`invalid runner request`, exit 1). It did not execute a
   source read: an independent guest check proved the operation directory and
   CSV destination absent. Failed command and operation identities are retained.
   `26495dd` sends only the explicit file/size/hash/URL wire fields. A GET-only
   reconciliation of the exact pre-execution rejection retains the failed
   attempt and makes the file eligible for another separately approved import;
   ambiguous or post-execution failures are never reset/replayed this way.
   All 120 unit tests, packaging and all six CI jobs (`33967456422`) passed.
   Installed extension bundle SHA-256:
    `e430a70648c5ba5a388280197ec30cbc83ef1382fea8ffdf82b7d43b10b23baa`.
3. Additional macOS runner regression tests exposed the test fixture's
   `/var` versus `/private/var` temporary-directory alias. Canonicalized only
   the fixture root, preserving production symlink-escape protection.
   `go test ./internal/runner` subsequently passed on this Mac.
4. After approximately one hour, ARM reconciliation returned HTTP 401. The
   installed azureauth implementation captures the original authentication
   session inside `AzureSubscription.credential.getToken`; it never refreshes
   that token. `efc1d99` obtains a fresh same-account, tenant-bound silent session
   for every ARM request and pagination page. It does not automatically replay
   failed writes. All 122 unit tests, packaging and CI `33969868871` passed.
   The updated extension is installed in VS Code 1.136.1; bundle SHA-256
   `3bcf9192c166f7f8f0f22bd3179aa70782f02ae8ac5493856e5f086d474a8900`.
   During refresh, Microsoft Authentication also reported `AADSTS50079` and
   `invalid_grant/basic_action`: interactive MFA is required and silent
   acquisition is not allowed. macOS broker/keychain errors were logged too.
   No alternate credential is substituted for the GUI qualification.

## September 5 historical pause and recovery boundary

Eight of 18 CSV imports have GUI-verified full-hash receipts. `Location.csv`
was submitted as the ninth import; its status reconciliation encountered the
authentication boundary. The other nine remain uploaded, not imported. All
18 upload receipts and all entered mappings are preserved.

At the first pause, the retained command was a **status-only** request, not a new CSV import:
`af-ca9d6344-00df-43bb-9391-da42f8185805`, operation
`f8929635-7ef9-4d98-be50-aac91008f6dd`. It is marked `unknown`; an independent
ARM GET returned 404. Do not reset or replay the import because of that absence.
After the user completes VS Code Azure authentication, reconcile the retained
guest operation and its seal before continuing. A safe recovery control for an
absent status-only ARM command may be needed; the persisted workflow must not
be manually rewritten to claim verification.

No source assessment or report transfer has started. No target deployment,
resize, migration or post-migration verification is claimed.

## Independent pause-time guest readback

At **13:50:15Z**, a read-only guest diagnostic found nine sealed CSV files,
522,291,068 bytes. Independently recomputed every byte's SHA-256 and compared
all nine file identities, sizes and hashes with the retained GUI manifests:
**9/9 matched**. This includes Location, whose GUI reconciliation is still
pending; the private workflow was not changed to claim it verified.

| CSV | Bytes | Guest SHA-256 matched desktop manifest |
| --- | ---: | --- |
| CARRIED_BY | 70,714,348 | `67d3328f0e2bcc9a9a2f1c71ad51edb8af43148d7abee5becab78c9d91fdee69` |
| CONTAINS | 204,764,876 | `00d11fffe89b4e5ce19c0046adbfb3fccb1791462a2ce7b3c2211fa347d61890` |
| Carrier | 235,855 | `0a2c6e8ecf1fdfe2540c4058202527e458f66e1d7611d75c03d51144089fe88b` |
| Customer | 6,371,355 | `4b82abd66754f08749e12d0f2cda3abbcfed46c3016855caf654a1608517a5bd` |
| DESTINED_FOR | 71,610,098 | `2da3ab71851271c5e2895d2ca2b2f9599af5334ae87ed431d5d09f4baa5bcec2` |
| FULFILLS | 92,615,438 | `0d8294d1cfa659300c6b3c275aca46757d93dd240ea504587d64420e70d67842` |
| Facility | 4,428,974 | `62d18a1c8beccfc9d63bf2d9c5a1b47bdc1b9bb7bc36ac2ceda5cfb2aca9de5e` |
| INCLUDED_IN | 69,294,950 | `f52ffd09f781a4cb1c0b435649371d79b0cd9b64f82883fb0fb8b08901f0d1a7` |
| Location | 2,255,174 | `4e4bdc6637d633ce6e1e63bcc6a0f2ba936edfd5cadfa532e2073809817d5a31` |

No active runner operation remained. Disk use was 5%
(2,854,469,632 / 66,404,147,200 bytes); available memory 7,780,790,272
bytes; swap, swap-in/out and OOM kills were zero. The task RG activity-log
check since 13:30Z returned no failed entries (not proof of every token-level
failure, which need not appear in that log). VM deallocation was requested
after the evidence check and independently confirmed `PowerState/deallocated`
before handoff. Disk, CSV files, storage and diagnostics are retained. NAT/IP
and storage continue to incur small charges; no resources were deleted.

## September 6 initial resume attempt (superseded by continuation below)

- At 01:21–01:29Z, rechecked the authorized 96-hour window, retained VM,
  unchanged storage access restrictions, 16:00 UTC shutdown and 25 USD trial
  reserve. Six overnight policy audit failures concerned the evaluator's GET
  permission on private endpoint connections; these were audit errors, not
  permission to change policy or a successful compliance result.
- `4636a70` adds GET-only reconciliation for a status-only ARM command absent
  for at least five minutes. Its identity is archived, its state is not marked
  successful, and a later explicit status request can read the existing guest
  operation. Imports, assessments, exports, readiness, recent and invalid-time
  requests are not cleared by this exception. All 124 unit tests, package build
  and CI `34003810189` passed. Installed and reloaded in the actual VS Code.
- Actual GUI reconciliation preserved the old status command in
  `absentStatusCommands` and released that status-only blockage, without editing
  the private workflow file or replaying the CSV import.
- A successful ARM GET was **not proof of full authentication recovery**. After
  restarting the same VM at 01:25Z, the GUI readiness request remained unknown:
  `af-3797f10a-8dc9-45c9-b0d2-51b22267ba54`, operation
  `f69f7a6c-7964-4554-b032-d1e31622def9`. The latest Microsoft Authentication
  log still reported `AADSTS50079`, `invalid_grant/basic_action` and prohibited
  silent acquisition. Retain this unknown request for reconciliation; do not
  replay it or claim a fresh GUI guest-readiness receipt.
- A separate read-only health check at 01:28:21Z confirmed cloud-init done,
  unchanged executable hashes, root-owned 0700 runner storage, disk 5%
  (2,864,205,824 / 66,404,147,200 bytes), available memory 7,796,563,968 bytes,
  no running AGEFreighter processes, zero swap and zero OOM kills.
- No additional CSV was imported and no source assessment started. The VM was
  again independently confirmed `PowerState/deallocated`, preserving disks and all evidence while
  the user completes interactive Azure authentication in VS Code.

## Retained GUI mapping review

Entered all nine vertex and nine edge mappings through the actual source-form
fields. Independently compared the persisted generated configuration against
`csv-source.json`: all labels, selected-file identities, ID columns, endpoint
labels/columns, property names and scalar/array types match exactly. This is
mapping evidence, not a successful migration. The September 6 comparison also
normalized implicit default string types before comparing effective types;
all 18 mappings and CSV defaults matched.

## September 6 completed bounded trial

- Actual GUI continuation reconciled Location's retained operation without
  re-importing it, then individually approved and verified the remaining nine
  imports. All **18 files / 1,168,576,671 bytes** are GUI-verified.
- Independent Linux full-byte readback at **02:12:02Z** matched all 18 sizes and
  SHA-256 hashes against the retained desktop manifests. See
  [machine-readable evidence](evidence/csv-guest-readback-20260906.json).
  Disk use was 6%, swap/OOM zero, and no active operation remained.
- Saved and pushed [24 retained command views](evidence/runner-commands-20260906-02.json)
  before a second capacity maintenance step. Fresh GET hashes matched the
  selected 21 successful terminal commands; only those ARM command resources
  were removed. Latest readiness/status and the initial decoder failure were
  retained, as were all guest directories, CSV seals and storage. No automatic
  housekeeping exists yet; the preflight prevents a blind 26th submission.
- A fresh GUI readiness check succeeded on the same boot
  `82ff617d-ed80-43c8-b5e5-13ae766ca158`. Reviewed all 18 source mappings and
  approved the read-only sampled profile through the native dialog at
  **02:14:19Z**. Operation: `0aa24eac-33c0-4747-9e44-65e50115c6cb`.
  Guest limits: 30 minutes / 4 GiB / no swap; CLI profile bounds included
  10,000 rows, 64 MiB and a two-minute source operation timeout.
- Worker exited successfully. Controller and guest configuration hashes were
  retained independently; the guest hash was stable through status/retrieval.
  [Command receipts](evidence/runner-assessment-commands-20260906.json) preserve
  acceptance, successful terminal state and the export result.
- Report export used the approved workflow storage destination. The GUI then
  reconciled the same export and imported exactly **26,514 bytes**, SHA-256
  `ed7e24eb327986d9574ae6e10581eadf06524c70d49879add29ea9cc47fc6409`.
  Independent local readback matched that manifest; file mode is 0600.
  The actual VS Code displayed “Hash-verified source report” with scripts
  disabled. Full source metadata remains private, not committed to this repo.
- **Report outcome: incomplete**, not a migration failure or whole-source pass.
  Observed 10,000 vertices, 0 edges, 3,302,577 sampled bytes; malformed/missing
  identity/endpoint/property signals were zero in that prefix. Mapping and
  read-only checks passed; source-read is unknown due to truncation and source
  version is unavailable for CSV. P1 has 5.6 million records, so these lower-bound
  observations must not determine whole-source capacity or migration acceptance.
- Final health at **02:17:20Z**: disk 6%
  (3,529,252,864 / 66,404,147,200 bytes), available memory 7,767,388,160 bytes;
  swap/OOM zero; no AGEFreighter process remained; approved binary hashes and
  root-owned 0700 directory unchanged. Deallocation requested at **02:18Z** and
  independently confirmed `PowerState/deallocated` before handoff.
  VM disk/data and report evidence are retained. NAT/IP/storage remain billable
  within the authorized 800 USD / 96-hour window; this is not infrastructure cleanup.
- Installed and built extension JavaScript match SHA-256
  `47c4a0d3a292f84e5f450819af845e955689235f05306d2344e0a30b68cd7b84`.
  The guest binary was not changed. No Marketplace release was published.

## Boundaries

The 18 P1 CSV files already passed actual GUI upload and independent full
readback on the desktop. This trial separately tests guest import/sealing and
bounded source assessment. Neither installation, ARM success nor a source
profile proves successful migration. All nine full migration cases remain open.
The task's 800 USD / 96-hour window ends 2026-09-09T08:55:00Z. This one-VM
trial uses a 25 USD reserve and must stop before its 16:00 UTC daily safeguard.
