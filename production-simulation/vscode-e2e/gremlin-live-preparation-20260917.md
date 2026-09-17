# Gremlin live-trial preparation checkpoint

September 17, 2026, approximately 07:42–07:50 UTC. **Preparation only; B05
and active-operation B10 are not qualified.** No Azure mutation, installed
extension replacement, source write, VM start or migration was performed.

## Fresh environment checks

- Correct authorized subscription selected. Nine surviving VMs are deallocated;
  all 17 Flexible Servers are Stopped. No resource-group lock was returned.
- The existing Cosmos account remains Japan East, public network Disabled and
  local/key authentication disabled. Database `p1` contains only the accepted
  `graph` container, partitioned on `/partitionKey`, autoscale maximum 4,000 RU/s.
  **Never replace or import Gremlin records into this accepted container.**
- Existing Cosmos assignments are Data Reader, including the retained
  preparation VM's identity. They do not authorize fixture creation. A future
  preparation grant must be reviewed separately, scoped only to a new container,
  and removed after successful preparation; account-wide write access is not
  needed. The guided migration identity remains read-only.
- The recent activity log shows external Storage, Defender and Event Grid
  operations around 07:34–07:35 UTC against the previous CSV trial's storage
  account. These are not our operations, and are not proof of a denial or cause
  of any failure. Do not undo them. Recheck transfer-account security and access
  before any future upload.
- Cost Management again returned HTTP 429. No fresh actual-billing claim is
  made. USD 800 / `2026-09-20T07:14:35.311Z` remain unchanged; the existing
  USD 600 planning reserve is not a current invoice or guaranteed headroom.
  Before creating the extra Cosmos container, include its incremental RU/storage
  retention, fixture preparation, runner/target compute and existing retained
  resources in the refreshed conservative cost gate.

## Local transfer capsule

Rechecked the fixed export manifest and SHA-256 of all 18 JSONL files:
5,600,000 records, 2,926,617,399 uncompressed document bytes. Source-side
canonical evidence remains the [previous full-P1 result](gremlin-offline-p1-20260917.md).

Retained capsule directory:
`production-simulation/work/gremlin-source-capsule.iUQDn9/`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `p1-gremlin-documents.tar.gz` | 246433349 | `f01f2044429b3a8cb2f1d123d0b2e41ec20bc1610c022e949ae5ec087219dcc6` |
| `cosmosfixtureload` | 11496336 | `fee91b06f436c4cd62f451a84c0c1a27a1509ddc46732c2d77f55d657b6e4b76` |

The archive contains exactly the 18 document files and `portable-manifest.json`,
without CSV copies. A complete streamed decompression matched the concatenated
original file bytes in archive order, SHA-256
`c102eaf5b90ec007e55db8c15683445dbc9c7838ed969fd00b7e3faecdc232ba`.
The Linux amd64 fixture loader was built with CGO disabled
from the clean `e2f8bdef0ae165a46b3e409086f1e29a7301f2ab` checkout. It is a
preparation-only managed-identity tool, not the migration runner or verifier.
It uses Upsert, so its existing historical preparation wrapper must **not** be
reused: that wrapper targets the accepted `p1/graph` and removes its local
document staging. A new wrapper must bind the new container, check emptiness,
verify the pinned archive and per-file hashes, use a fresh guest directory,
bound execution, and retain failure evidence without automatic replay.

An initial local tar invocation had incorrect BSD tar option ordering and
produced an incomplete archive in `gremlin-source-capsule.5K7T2d/`. That attempt
is retained and **must not be uploaded or used**. The table above identifies
only the corrected capsule.

## GUI and preservation

The signed-in installed VS Code is accessible. The existing terminal-state
Cosmos workflow is retained; no load/resume button was activated. The installed
bundle is still `34dc9fc71284d7a356aa52f40b03772a06197b40887dfd0ade79b4216707d91e`.
The new VSIX's SHA-256 was rechecked against the
[pinned candidate](gremlin-target-preflight-20260917.md#pinned-local-artifacts).
Action-time approval to install this unpublished candidate and reload the
window has been requested; the update has not yet been applied.

At `2026-09-17T07:46:53.591Z`, all 66 root workflow/report JSON files retain
aggregate SHA-256
`819cdcd7c1bae30b57cb669fcd0f7a2d45172c912072e05219f9197b33c28e9f`.

## Next execution sequence

1. After approval, install the exact pinned VSIX, reload, verify the installed
   bundle and preserved workflow state. Do not reuse an accepted workflow for
   the fresh Gremlin migration.
2. Refresh cost/time/security gates and prepare a separately named Cosmos
   container (proposed `p1/graph-gremlin-p1-20260917`). Retain the original
   container unchanged, keep private access, and explicitly bind any temporary
   preparation role and immutable transfer prefix to this trial only.
3. Prepare/verify all 5.6M source documents using a bounded, evidence-retaining
   fresh guest execution; remove its temporary writer before migration reads.
4. Run the installed-GUI source configuration, complete inventory, target
   review/deployment, same-VM resize, load and counts checks with the pinned
   runner. Capture an active-operation reload and GET-only reconciliation/no
   replay for B10, without causing an unplanned migration fault.
5. Use only the profile-bound Gremlin verifier for all 64 target ranges and
   canonical-root equality. Source preparation/counts alone cannot qualify B05.

No new budget window, target, role, network exception or cloud artifact was
created by this checkpoint.
