# B08 CSV interruption qualification

Status: implementation/local regression and installed network-disabled refusal
PASS; live interrupted transfer awaits the trial storage network decision.

## Change and review

The installed workflow previously offered non-cancellable upload progress. The
candidate now exposes Cancel for each file and forwards cancellation through
token acquisition, Blob requests and each 8 MiB block. It checks cancellation
before committing the block list. Hashing before approval remains a read-only,
non-cancellable inventory step. Cancellation never starts assessment or migration.

Cancellation retains earlier uploaded files and the current `prepared` manifest;
later files are not attempted. It does not delete uncommitted blocks or overwrite
committed blobs. An explicit retry needs the normal upload confirmation and
reuses the content-addressed destination. A lost commit acknowledgement, including
a cancel racing with a successful commit, remains uncertain until that retry
checks remote metadata and rehashes the local bytes. Metadata is not guest
verification. Changed local manifests are rejected, not silently replaced.

Review covered cancellation before token/network access, mid-block cancellation,
late commit acknowledgement, listener disposal, preserved workflow identity and
redacted diagnostics. The original timeout still bounds requests when no cancel
is requested. Existing development-artifact transfer behavior is unchanged.

## Local evidence

- TypeScript typecheck: PASS.
- Unit suite: **235/235 PASS**, including five new transfer tests and three panel
  controller cases (two cancellation/acknowledgement variants and changed retry).
- Build and host-test compilation: PASS.
- Isolated VS Code **1.138.0** Extension Host: **13/13 PASS**, exit 0, at
  `2026-09-16T13:15:10Z`. This profile has no signed-in user credentials.

The transport/controller tests use inert adapters. They prove neither a live
Azure fault nor the installed Cancel interaction; B08 remains partial.

## Installed candidate and live preparation

Code commit: `a459140`. The candidate was packaged, installed and reloaded in
the signed-in VS Code 1.138.0 profile. Built and installed bundle SHA-256 match:
`eb453a1bca36ec7a9f6e1b3dd3cb5c9cc9bf2b78d2cc63c6d74b1dcbdc9243c7`.

The GUI created independent draft `19a5ce10-4de7-4b3d-9104-7629ce18804c`:
CSV/local, approved trial subscription/group, Japan East/zone 1. Three separate
copies of frozen P1 Supplier, Lot and Shipment CSVs were selected through its
native folder dialog (193,209,631 bytes total). No accepted fixture was edited.
All previous 64 workflow/report JSON files retain aggregate SHA-256
`917a29e7457ef2b6f235970141cdc4c262c10543329aa04d189e7c18ee4fc8f8`.

After the user's account-specific approval, the installed native confirmation
created `af19a5ce104de74b3d910476` with the account-scoped user Blob Data
Contributor role. Deployment succeeded, shared keys/anonymous access remain
disabled and minimum TLS is 1.2. Fresh ARM read and GUI reconciliation both
observe public network access **Disabled**, despite the reviewed creation
template requesting Enabled. The precise actor is not attributed: the immediate
filtered activity-log query returned no entries yet. No automatic override was
attempted.

The actual installed upload action (three reviewed copies) rejects this network
state with the explicit governance/private-connectivity diagnostic, not a false
Azure sign-in error. The saved draft has no prepared/uploaded transfers,
assessment or migration. This proves the network-disabled refusal, not a
successful transfer or injected interruption. A separate user decision was
requested for the previously discussed trial-only exception on this new
account; it has not been applied here.

Fresh inventory before preparation confirms all eight surviving VMs deallocated
and all 17 Flexible Servers stopped. No VM/target was created or restarted.
Retained storage/Cosmos charges continue. The USD600 planning reserve is not an
actual-billing measurement; this subtest adds only storage/request charges for
about 184 MiB. The RG's older expiry tag still reads September 16; the explicit
user extension to September 20 is authoritative for this test, not permission
to remove external governance or silently update old resource tags.

## Next live steps and constraints

Use a new CSV-only workflow and copies of frozen P1 files. Do not change accepted
source paths, manifests, blobs, graphs or reports. Start with transfer-only work:
no VM or Flexible Server restart is required. Inspect current ownership, budget,
deadline and governance before provisioning a workflow-owned storage account.
Any new scoped access grant is separately confirmed at the actual approval
surface. The outer deadline remains September 20 16:14 JST and the ceiling USD800.

Capture actual Cancel, retained prepared state, explicit retry/reconciliation,
changed-file rejection and no assessment/load before complete guest receipts.
Do not count simulated HTTP responses as live evidence. Retain aborted blocks
and all trial evidence; do not fault the already qualified CSV workflow.
