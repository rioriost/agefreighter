# Gremlin installed-GUI migration — fresh draft

September 18, 2026 JST (September 17 21:42–21:48 UTC). **Draft prepared;
storage approval pending; no migration or new Azure deployment submitted.**

The [source preparation](gremlin-source-execution-20260917.md) passed separately.
Its 5.6M Gremlin-shaped NoSQL documents are not a GUI/target qualification.

## Fresh gates

- USD 800 and `2026-09-20T07:14:35.311Z` remain unchanged. Billing refresh
  returned 429; do not label a stale billed total current. Keep the existing
  USD 600 extended-retention reserve and the separately reserved preparation
  increment. Runner/target execution requires a bounded incremental estimate.
- All nine VMs are deallocated and all 17 Flexible Servers are Stopped. No RG
  lock was returned. Cosmos remains private with local/key auth disabled; the
  isolated container still uses `/partitionKey`. Its temporary writer is absent.
- An external disk-write activity at September 17 15:56 UTC targeted the retained
  preparation OS disk. The disk still exists, Reserved/Succeeded, 64 GiB. No
  restart, modification or attribution was inferred from that event.
- The installed extension bundle remains
  `073232d2528ed59271ed33d5a43dba9e77235a4a5757e1555fefdee1b980e618`.
  Both [pinned runner/verifier archives](gremlin-target-preflight-20260917.md#pinned-local-artifacts)
  retain their recorded hashes. Neither was rebuilt or dispatched.

## Actual GUI steps

The installed VS Code opened a **new** guided workflow, without a project-folder
prompt or desktop CLI. Its existing Azure session loaded the authorized
subscription and listed the dedicated resource group. Cosmos discovery returned
the expected `afcosmosp120260907` candidate. Selected Japan East, zone 1,
B2s_v2 and the existing `vnet-af-vscode-p1/runner` subnet.

The initial paste did not populate the subnet field; preflight rejected the
empty value. After setting and visibly verifying the full ARM ID, normal release
preflight correctly refused the unpublished AGEFreighter 2.4.0 Linux release.
No deployment was submitted. Continue via the existing explicitly gated pinned
development-artifact path, not by weakening release verification.

New local draft: `4043e008-b86e-47b8-8722-1efe637ae12a`.
In its source form, reviewed migration name `p1-gremlin-20260918`, namespace
`migration`, host `afcosmosp120260907.documents.azure.com`, database `p1`,
container `graph-gremlin-p1-20260917`, Gremlin-over-NoSQL format,
`partitionKey`, and `score=float64,distance_km=float64`. No source assessment
has started. The accepted explicit-document workflow was not reused.

The native approval dialog is open for **new** transfer account
`af4043e008b86e47b887221e`, Japan East / Standard LRS, with a Blob Data Contributor
grant to the signed-in user on that new account only. Anonymous/shared-key access
is disabled; the HTTPS endpoint is network-public, not a private endpoint. A
specific action-time approval was requested before creating the access grant.
`storageDeployment` is still absent; no account/role was created by this turn.

## Next gates

After approval, create/reconcile this workflow's transfer storage and validate
its actual access. Prepare the pinned `e70e02068c68` development runner archive,
then review its new VM and exact Blob Reader grant. Review the Cosmos Data Reader
scope before granting it. Obtain complete inventory, privately deploy/review the
target, resize the same runner, load, verify counts, and compare all 64 target
ranges with the Gremlin root. Include active-operation reload/no-replay evidence
for B10. Each approval remains bound to its actual artifact/resource/scope.
B05/B10 remain open; this checkpoint is not source assessment or migration PASS.
