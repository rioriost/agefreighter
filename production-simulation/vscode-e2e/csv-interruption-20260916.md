# B08 CSV interruption qualification

Status: installed network-disabled refusal, upload cancellation, changed-file
refusal and explicit retry with full Blob readback **PASS**. B08 remains partial
for the separate guest-import/hash-mismatch and lost-commit-acknowledgement cases.

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

## Authorized exception and installed transfer results

The user explicitly requested `SecurityControl=Ignore` on this exact account,
then separately approved enabling its public endpoint and resuming this trial.
Both changes were narrowly applied and re-read from ARM. Public access is
Enabled, HTTPS required, anonymous access and shared keys remain disabled;
minimum TLS remains 1.2. No other account or source firewall was changed.

The first three frozen P1 copies uploaded successfully, but the transfer was too
fast to cancel. That attempt is **not** counted as interruption evidence. To
make a real native Cancel reproducible, the GUI selected two additional files:
one header plus eight repetitions of the frozen CONTAINS rows (1,638,118,385
bytes), and a Supplier copy as the trailing file. These are **transport-only**
fixtures with intentional duplicate identities; never import/migrate them or
claim them as another P1 graph qualification. The local generator uses exclusive
file creation and enforces the 2 GiB per-file bound.

On the next explicit upload, the actual VS Code notification showed the large
file at 15%; Cancel was then clicked. The panel reported cancellation and no
automatic retry/migration. A subsequent Blob BlockList GET proved **0 committed
blocks and 57 uncommitted 8 MiB blocks** (478,150,656 bytes). The percentage at
the observation is not the exact cancellation boundary: transfer continued
between observing and clicking. Durable state retained the original three
`uploaded` entries, the large file `prepared`, and no transfer entry for the
trailing file. There was no assessment or migration.

The independent Supplier copy's header was then deliberately changed (not the
frozen source); SHA-256 became
`10c62d8063a416f3bfc80cf8732231bea01be0ef12fa342e94723daf93185287`.
An installed-GUI explicit retry refused it with "A previously reviewed CSV
changed". Listing the committed blobs showed only the original three, with
unchanged ETags. The trial copy was restored and its SHA-256 again matched the
frozen Supplier file. No saved workflow JSON was edited to manufacture a result.

A fresh native upload confirmation then retried the **same** large-file UUID,
size, digest and content-addressed destination, followed by the previously
unattempted trailing file. All five entries became `uploaded`. The panel cleared
its old error and still displayed "No assessment started"/"No report transferred".
This retry re-sends the bounded blocks; it is not byte-offset resume.

Independent authenticated GETs streamed and hashed **all 1,840,125,623 bytes**;
all five hashes and sizes matched the reviewed manifests, finishing at
`2026-09-16T13:36:17.109Z`. No sensitive token or file contents were retained in
the public evidence. The original three ETags were unchanged. The earlier 64
workflow/report JSON files also retain the exact aggregate hash above.
See [structured evidence](evidence/csv-interruption-20260916.json).

Final read-only Azure inventory again confirms **8/8 VMs deallocated** and
**17/17 Flexible Servers stopped**. Storage/evidence are retained; no cleanup
or source/target mutation was performed by this transfer trial.

## Remaining live steps and constraints

Use a new CSV-only workflow and copies of frozen P1 files. Do not change accepted
source paths, manifests, blobs, graphs or reports. Start with transfer-only work:
no VM or Flexible Server restart is required. Inspect current ownership, budget,
deadline and governance before provisioning a workflow-owned storage account.
Any new scoped access grant is separately confirmed at the actual approval
surface. The outer deadline remains September 20 16:14 JST and the ceiling USD800.

The desktop cancellation/retry and changed-manifest cases above are complete.
Do not confuse Blob readback with Linux import sealing or graph verification.
The guest receipt/hash-mismatch gate still needs its own live case using a
valid isolated source. A lost *committed* acknowledgement currently has unit
evidence, not an injected installed-GUI result. Preserve all evidence and the
transport-only fixture; do not fault the already qualified CSV workflow.
