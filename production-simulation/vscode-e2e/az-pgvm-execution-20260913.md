# AZ-PGVM guided P1 execution

Status: GUI source configuration reviewed and saved; awaiting the dedicated
storage/scoped-role approval. This path is not qualified.

## Retained setup

- Installed VS Code 1.136.1, AGEFreighter extension 2.4.0.
- Workflow: `2595fb2d-df9d-4582-b8b8-237ae211ec1c`.
- Source selected through the subscription/resource-group/Azure VM discovery UI.
- PostgreSQL 18 fixture, database `p1source`, read-only source role; TLS
  certificate validation remains required with the fixture's custom CA.
- All nine vertex and nine edge mappings were entered and reviewed in the GUI.
  Stable IDs, endpoints and typed source properties match the prepared fixture.
- Discovery runner proposed in Japan East, zone 1, existing runner subnet,
  `Standard_B2s_v2`; no runner or target has been deployed for this workflow.
- Frozen local development runner built from
  `2fd3aa4c157fb1e03107922b6b16b47a1b5a97fe`, archive 37,040,125 bytes,
  SHA-256 `df8b6244963bd059389b3057274392c64164118dad0ca5949e69b04606cfa8fb`.
  It has not been uploaded or deployed.

## Safety and UI observations

All six existing trial VMs were deallocated at the initial live check, and no
resource-group locks were present. Recent external configuration writes attempted
to require TLS/set its minimum version on the retained Flexible Servers; they
failed with `ServerIsBusy`. No change to those resources was attempted here.
Before starting compute, refresh cost, deadline, governance and resource-state
gates under the renewed USD 800 / 96-hour authorization.

The native clipboard-based input path produced unexpected oversized values in
several unsaved fields. They were replaced using direct accessible-field value
entry before any review/save or source request. The final saved source draft
contains only the intended connection fields and all 18 mappings.

Selecting a custom CA reinitializes the form and discards unreviewed mappings.
The certificate was therefore selected first, all mappings re-entered, and the
reviewed persisted draft checked. Preserve this as a usability defect to fix;
do not confuse successful field entry with persistence.

The current GUI approval creates workflow-owned Standard LRS storage and grants
the signed-in user Storage Blob Data Contributor on that new account only.
Its HTTPS endpoint is network-public; anonymous access and shared keys remain
disabled. It does not expose the source server. The approval has not been
accepted and no cloud mutations were made in this step.

## Remaining qualification

Approve and reconcile storage; prepare the pinned runner; deploy and verify its
guest; start the source only after safety gates; run complete mapped inventory;
review capacity and private target deployment; resize the same runner; migrate;
verify exact counts and all 5,600,000 records across 64 canonical ranges; retain
evidence and stop compute. Headless preparation is not GUI qualification.
