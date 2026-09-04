# VS Code guided migration contract

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
| `profile` | validated source fields, one password, placement fields | Store the password, write a protected draft, validate it, read exact Neo4j count-store totals, and run a bounded read-only profile. |

The extension replies with `subscriptions`, `azureSignedOut`, `busy`, `error`,
or `profileComplete`. Replies never contain a credential, authentication token,
connection string, source query, or source record. Password controls are cleared
immediately after a `profile` message is posted and are never included in
`vscode.setState`.

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

The CLI profile's `complete-stream-range` may be used as a sizing input. For a
Neo4j discovery job, `inventory` reads exact unfiltered node and relationship
totals using simple transactional count-store queries. A bounded profile may be
scaled by those totals because the guided draft migrates the whole database;
the proposal records that method and retains the profile's estimation range.
Without trustworthy totals, the extension must not silently extrapolate a
bounded prefix.
