# AZ-PGVM guided P1 execution

Status: complete GUI source inventory passed and its hash-verified report is
imported. Private target review is next. This path is not qualified.

## Retained setup

- Installed VS Code 1.136.1, AGEFreighter extension 2.4.0.
- Workflow: `2595fb2d-df9d-4582-b8b8-237ae211ec1c`.
- Source selected through the subscription/resource-group/Azure VM discovery UI.
- PostgreSQL 18 fixture, database `p1source`, read-only source role; TLS
  certificate validation remains required with the fixture's custom CA.
- All nine vertex and nine edge mappings were entered and reviewed in the GUI.
  Stable IDs, endpoints and typed source properties match the prepared fixture.
- Discovery runner proposed in Japan East, zone 1, existing runner subnet,
  `Standard_B2s_v2`; runner is deployed, target is not yet created.
- Frozen local development runner built from
  `2fd3aa4c157fb1e03107922b6b16b47a1b5a97fe`, archive 37,040,125 bytes,
  SHA-256 `df8b6244963bd059389b3057274392c64164118dad0ca5949e69b04606cfa8fb`.
  It was uploaded through the GUI and the guest verified this exact artifact.

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

Two CA persistence defects were found before source reads: preview creation
dropped the CA binding, and selecting a CA reset unsaved mappings. Both were
fixed in `c1bd344`, with regression tests. All 165 extension tests, typechecking,
compilation and packaging passed. The updated bundle was installed in VS Code
1.136.1 and the window reloaded while no source operation was active. GUI
re-selection now preserves all 18 mappings and binds the reviewed CA hash.

The user approved storage creation. At 12:23–12:27Z, read-only reconciliation
confirmed the workflow-owned account is `Succeeded` and the signed-in user has
Storage Blob Data Contributor scoped to that account only. Azure activity
records show successful `policies/modify/action` events during creation;
the resulting `publicNetworkAccess` is `Disabled`. Anonymous access and shared
keys are also disabled. The GUI reports storage ready but explicitly says
provisioning is not transfer readiness. No artifact upload was attempted.

After explicit user approval, only the new workflow storage received the
organizational `SecurityControl=Ignore` tag and enabled authenticated HTTPS
public networking. Anonymous access and shared keys remain disabled. The
approved pinned archive was uploaded and the discovery VM deployed. Source
access remains private; the source VM and this runner are running. The other
five VMs remain deallocated. An old CSV Flexible Server found running was
stopped; all four retained Flexible Servers are stopped.

The first readiness probe preceded cloud-init completion and failed without
replaying bootstrap. Read-only inspection subsequently proved cloud-init done,
no bootstrap errors and both binaries present. A fresh GUI readiness check at
12:50:55Z passed: runner idle, disk below 4%, zero swap/OOM, pinned artifact
unchanged. Source preflight confirmed the retained 18-table fixture, read-only
TLS access, disk 9%, no swap/OOM and no public source IP.

Resource-group Cost Management succeeded after subscription-scoped throttling:
September 12/13 billed daily amounts were USD 17.515729 and 2.419002 (billing
lags; not a final total). The renewed USD 800 ceiling and
2026-09-16T07:14:35.311Z deadline remain unchanged.

The GUI approved complete source reads, using the existing read-only credential
only through its private input and protected guest channel. Inventory operation
`d207e6cf-f91c-43bf-ae44-44801a18148b` was accepted but failed in 18 ms at
12:51:46Z. The source configuration hash
is `1a2d31da1ab84c049962a3fa92c0654dfc57ddb05961173f3b64643d0365fda5`;
custom CA SHA-256 is
`0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
No target writes or migration qualification have occurred.

The 51-byte guest error is `inventory: network inventory initialization failed`;
its SHA-256 is
`d1d694a1717436e62073c3f131686fba9f5f11455eb128dd409544982872d94b`.
Transient secrets were removed. DNS resolves correctly and the source logged
no authentication, HBA or SSL-accept rejection in the operation interval.
Live certificate inspection matches the retained certificate: validity ended
2026-09-09T23:26:14Z. The CA itself remains valid through October 6.

Commit `995ca57` adds a separately confirmed failed-assessment retention action.
It requires a fresh same-boot idle guest, no swap/OOM, disk below 80%, no target
or migration, and retains the old operation in history. It never resumes or
automatically retries anything. All 167 extension tests and packaging passed;
the installed bundle matches the build and VS Code was reloaded.

Certificate renewal preserves the existing CA, private key, names and data.
The renewed public certificate is valid through September 20 and has SHA-256
`0be89b41afc0cfb0afbb27d47befb831a4e076db5df52203b1a94af218a0fbef`.
The source VM reached Azure's 25 managed-command resource limit. Its completed
read-only authentication diagnostic definition and result were archived locally
with SHA-256 `58bdd1a614dd7bfeb219a274ee3256450604fb686fe122e34d3eafbfba06a0ed`
before removing only that command resource to free a slot. No database, disk,
guest operation directory or failed inventory evidence was deleted. The prior
certificate-check result was also archived (SHA-256
`06c7b88246389e61e3cce5a86d0b6930a263c07323c030369e756b1006a1508c`).

Renewal succeeded and proved `source_container_was_running=false`: starting the
VM had not started its restart-disabled database container. The same container
was started after renewing the certificate, without changing the database,
private key, CA or network. The successful renewal evidence has SHA-256
`a36bb7e901dc1c43070668526558f9eb6be4b4dccf043846b7c29e551c28ce47`.
The preflight script now requires a running container and a certificate valid
for the next 96 hours, rather than treating retained preparation-time TLS
evidence as current connectivity proof.

The GUI retained the failed operation in history, reviewed unchanged source
settings and explicitly started new inventory
`1ae33ad6-789f-4705-ac91-af36b60ae6aa`. It passed at 13:25:04Z: all 1,600,000
vertices, 4,000,000 edges and 18 exact label counts, zero errors, one complete
repeatable-read stream. Decoded mapped-record bytes are 458,398,000; the sizing
range is 3,008,790,000–8,567,972,000 bytes before additional target headroom.
The 2,944-byte report was exported and hash-verified in the installed GUI:
`044b34352a83bbdca3bfd8a257da3162a5f3f8ece9a61634434d3750643e3b02`.
Five completed runner diagnostic command resources were likewise archived with
checksums before removing only their Azure management entries, freeing slots
for migration/verification. Their guest operation data and all source/target
data remain retained.

## Remaining qualification

Review capacity and private target deployment; resize the same runner; migrate;
verify exact counts and all 5,600,000 records across 64 canonical ranges; retain
evidence and stop compute. Headless preparation is not GUI qualification.
