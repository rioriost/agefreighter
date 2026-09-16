# B08 CSV interruption qualification

Status: implementation and local regression PASS; live interrupted transfer pending.

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
