# VS Code guided migration contract

> Historical version-1 local profiling contract. The active wizard now uses the
> [version-2 runner-first contract](vscode-runner-contract.md). Existing v1 drafts
> are retained, not reinterpreted as deployable runner state.

## Current routing (2.4.0)

The active source selector supports CSV, Neo4j, PostgreSQL and Cosmos DB for
NoSQL. It opens without a workspace or desktop CLI, stores version-2 workflow
records in private extension global storage, and runs approved operations on a
pinned Linux Azure runner. Source passwords use native protected prompts and
optional scoped SecretStorage, not the historical webview password message.
Target review creates private PostgreSQL 18/AGE with HA disabled; explicit
resize, migration, same-job recovery and sealed verification are implemented.

Use the [current operating guide](vscode-extension.md) and
[runner contract](vscode-runner-contract.md) for executable behavior. The
[qualification ledger](../../production-simulation/vscode-e2e/remaining-validation.md)
records 9/9 defined base routes and 12/12 finite branches PASS. M6 release checks
and actual publication remain separate. The following v1 actions, workspace
paths, 24-hour proposal and desktop CLI prerequisites are historical protocol
facts, not options exposed by the current wizard.

## Historical version-1 contract

This contract supplements the [VS Code CLI contract](vscode-extension-cli-contract.md)
for the 2.4.0 guided migration workspace.

## Trust and authentication

- A local, trusted VS Code workspace is required.
- The operator is signed in to Azure through VS Code's built-in Microsoft
  authentication provider. The extension respects the subscriptions selected
  in the Azure Resources subscription filter.
- Azure Resource Manager tokens remain in the extension host. They are not
  written to files, messages, logs, telemetry, reports, or model context.
- The deprecated Azure Account extension and an independent `az login` session
  are not used.

## Webview messages

The webview may send only these version-1 actions:

| Action | Payload | Effect |
|---|---|---|
| `ready` | none | Read the VS Code Azure session and selected subscriptions. |
| `listRegions` | selected subscription ID | Read available Azure regions for the on-premises recommendation field. |
| `profile` | validated source fields, one password, placement fields | Store the password, write a protected draft, validate it, read exact Neo4j count-store totals, and run a bounded read-only profile. |

The extension replies with `subscriptions`, `locations`, `azureSignedOut`,
`busy`, `error`, or `profileComplete`. Replies never contain a credential,
authentication token, connection string, source query, or source record.
Password controls are cleared immediately after a `profile` message is posted
and are never included in `vscode.setState`.

Unknown actions and unexpected fields have no effect. Mutating Azure and
migration operations will use separate typed messages with modal confirmation;
they are not part of the M1 contract.

## Local artifacts

The extension writes draft state below `.agefreighter/guided/<workflow-id>`.
This directory is excluded from source control and normal LoadJob discovery.
The state format is defined by
[`guided-migration-state.schema.json`](guided-migration-state.schema.json).

Source credentials are stored in VS Code SecretStorage. The CLI also needs a
stable file reference, so the extension materializes an owner-only copy below
its `globalStorageUri`. The LoadJob contains only that absolute file path. On
Unix hosts the directory and file modes are enforced as `0700` and `0600`.

## Placement evidence

An Azure source is considered verified only when the selected ARM resource is
found in the selected subscription and its returned location and single logical
zone are recorded. A hostname is never location evidence. For an on-premises
source, the state records a user-declared physical location; a later region
recommendation must retain `declared` confidence until the user approves it.

## Capacity evidence

The CLI's `complete-stream-range` may be used as a sizing input only with a
passing report, no failed/unknown checks, and complete positive row coverage.
For a Neo4j discovery job, `inventory` reads exact unfiltered node and relationship
totals using transactional count-store queries. CSV inventory scans every
configured mapping and records complete mapped counts and record widths under
explicit limits; the remote path requires the `csv-inventory-v1` guest capability.
A bounded prefix may be scaled by exact totals for review, but this does not make
it representative or deployable. Combining counts with complete-stream capacity
preserves eligibility only when their totals match exactly. Storage multipliers
remain estimates, not deployment approval or migration-verification evidence.

## Azure proposal evidence

The extension reads PostgreSQL location capabilities and quota, Compute SKU
availability and quota, and USD Retail Prices at runtime through the selected
VS Code Azure subscription. It writes the bounded result as
`azure-proposal.json`, conforming to
[`azure-deployment-proposal.schema.json`](azure-deployment-proposal.schema.json).
The proposal expires after 24 hours and does not authorize deployment.

For an Azure source, its verified region and logical zone are the defaults. For
an on-premises source, Azure's physical-location metadata may preselect a region
only when the text match is unambiguous; the operator confirms or changes it.
The target Flexible Server and loader VM must share a supported logical zone.

Predeployment AGE support is advisory service-matrix evidence. Migration remains
blocked after deployment until the actual server proves the AGE allowlist,
preload setting, and `pg_available_extensions` entry.
