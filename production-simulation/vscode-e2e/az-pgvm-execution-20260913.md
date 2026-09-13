# AZ-PGVM guided P1 execution

Status: GUI source configuration saved and dedicated storage/scoped role created;
transfer blocked by Azure Policy network modification. Development artifact
approval is displayed but not accepted. This path is not qualified.

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

The user approved storage creation. At 12:23–12:27Z, read-only reconciliation
confirmed the workflow-owned account is `Succeeded` and the signed-in user has
Storage Blob Data Contributor scoped to that account only. Azure activity
records show successful `policies/modify/action` events during creation;
the resulting `publicNetworkAccess` is `Disabled`. Anonymous access and shared
keys are also disabled. The GUI reports storage ready but explicitly says
provisioning is not transfer readiness. No artifact upload was attempted.

The five older trial storage accounts have the user-approved organizational
`SecurityControl=Ignore` tag and enabled public networking. The new account has
neither. Applying that exemption and enabling authenticated HTTPS on this exact
account is awaiting confirmation; no policy, tag or network setting has been
changed by this continuation. Source access remains private.

The GUI has inspected and hash-verified the retained development archive and
displays its pinned-artifact approval. No new executable has been uploaded or
deployed. All six existing VMs remain deallocated. No new runner/target exists.

## Remaining qualification

Approve and reconcile storage; prepare the pinned runner; deploy and verify its
guest; start the source only after safety gates; run complete mapped inventory;
review capacity and private target deployment; resize the same runner; migrate;
verify exact counts and all 5,600,000 records across 64 canonical ranges; retain
evidence and stop compute. Headless preparation is not GUI qualification.
