# Source-location and CSV-cancellation GUI audit

September 16, 2026, approximately 12:59–13:05 UTC. Scope: B03 and the
selection-cancellation part of B08; not a new migration or cloud fault trial.

## Installed GUI evidence

The normal Azure-signed-in Mac profile ran VS Code 1.138.0, build
`7debcd0e2acdea1c52de81bf9ee1620444407dda`, arm64. Installed AGEFreighter
2.4.0 bundle SHA-256 remained
`f1468e18046b352d93706d8123a6c8b6a460751190f6ab7b0646e2033de885ea`.
No extension installation, credential access, or cloud write was performed.

The existing migration wizard was closed before opening a genuinely new one.
The new wizard opened without requesting a project folder or local CLI, had
no selected saved job, and disabled execution/readiness controls.
Native dropdowns and the accessibility tree were inspected after each choice:

| Source | Available locations | Observed behavior |
|---|---|---|
| Neo4j | azure / on-premises / other-cloud | Azure candidate and source ARM fields appear only for Azure. Both endpoint-only selections hide them and describe existing private connectivity. |
| PostgreSQL | azure / on-premises / other-cloud | Same observed selection behavior; no source ARM field for either endpoint-only selection. |
| Cosmos DB for NoSQL | azure only | Azure candidate/source fields remain; no on-premises, other-cloud or local option. |
| CSV | local only | Local file picker appears; selecting CSV alone does not open a project-folder dialog or upload files. |

Three actual native file-dialog cancellation checks passed:

1. New wizard → CSV → Select local CSV files → Cancel. No selected files,
   saved workflow, deployment, readiness check or migration was created.
2. Explicitly reconnect to accepted CSV recovery workflow
   `54da6ddd-27d2-45e0-bb68-cf5f352801db` → Configure source → Select CSV
   folder → Cancel. All 18 selected mappings remain visible.
3. The same source editor → Select local CSV files → Cancel. All 18 mappings
   remain; no error, upload or import is triggered.

Before/after SHA-256 comparison found all **64** retained JSON files unchanged
(16 workflow records plus 48 reports), with no new files. Aggregate SHA-256:
`917a29e7457ef2b6f235970141cdc4c262c10543329aa04d189e7c18ee4fc8f8`.
The aggregate hashes sorted filename + NUL + per-file lowercase SHA-256 + LF.
This is evidence for these files, not all VS Code application storage.
No accepted fixture, graph, report or credential was edited to manufacture
this result. The accepted source editor was inspected but not saved.

## Automated regression evidence

Seven additional tests exercise production logic with inert adapters:

- Four Neo4j/PostgreSQL × on-premises/other-cloud preflight cases record every
  request/list call. Only runner subnet, VNet, resource group, SKU and quota
  are read: no source VM/database identity or source resource enumeration.
- Two cases prove identical LoadJob configuration for on-premises and
  other-cloud selections with otherwise identical Neo4j/PostgreSQL input.
- One production new-wizard message-handler case proves canceled CSV selection
  neither persists a record, invokes cloud/guest controls nor emits a selection.

TypeScript typecheck and **227/227 unit tests PASS**. The real current Go CLI
validator accepts all **9/9** generated source configurations, including both
endpoint-only locations and both Cosmos formats. The first contract invocation
omitted the required test-binary environment variable and failed before running
the validator. A fresh temporary local Go build plus the explicit binary path
resolved that harness setup error; it did not change the desktop guided flow.

The request traces are local regression evidence, not live network packet
capture. The actual GUI source-selection audit and the previously accepted
on-premises P1 routes complement them; no new external-cloud migration is
claimed. Tests changed only; production code and the installed bundle did not.

## Preservation and remaining work

Read-only Azure inventory during the audit confirms all **8** surviving trial
VMs deallocated and all **17** Flexible Servers stopped. Budget USD 800 and
September 20 07:14:35.311 UTC outer deadline are unchanged. Storage/Cosmos
charges continue.

B08 is now partial rather than not-run: its native selection-cancellation
checks pass, but interrupted multi-file upload, changed-file rejection and
reconciliation still need installed-GUI evidence. B03's selectable-location
audit is complete; no additional other-cloud end-to-end service qualification
is claimed. B05 Gremlin and other open ledger cases remain distinct.
