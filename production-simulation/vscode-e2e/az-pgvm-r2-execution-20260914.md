# AZ-PGVM corrective GUI attempt

Status: corrected source draft reviewed; authenticated artifact transfer and
private Linux runner readiness and source health passed. The installed GUI is
waiting for the read-only source password before complete inventory. No target
deployment or migration has started.

## Preserved original

Original workflow `2595fb2d-df9d-4582-b8b8-237ae211ec1c`, migration job
`12a2462e-e5a3-4368-a356-54e292650051` and failed verifier
`c0efdd7b-fd18-49db-a872-bbd29d1736c2` remain unchanged. Their graph, source
data, configuration, reports and guest evidence are retained. The failed
verifier's active marker was not cleared. See the
[first execution sheet](az-pgvm-execution-20260913.md).

At 2026-09-14 00:33 UTC, the installed GUI selected the original target and
the full-P1 reconcile action. On the deallocated VM, the current ARM response
reports instanceView `Pending`, exitCode 0 and provisioningState `Succeeded`,
without the previous execution output. These do not supersede the retained
exit-1 failure evidence or prove success. The GUI keeps its local phase
`submitted`; no operation was replayed and no private state was manually patched.
Do not restart old compute merely to update this display.

## New source draft

- Workflow: `22f11b89-e943-4d56-9675-7331a78b6de7`.
- Name / proposed graph: `az-pgvm-p1-r2` / `az_pgvm_p1_r2`.
- Azure resource discovery selected the existing PostgreSQL source VM.
- Existing trial resource group, Japan East / zone 1 / existing runner subnet.
- Discovery SKU: `Standard_B2s_v2`; private VM provisioned on September 14.
- PostgreSQL 18, database `p1source`, existing read-only role and validated TLS.
- Selected public CA SHA-256:
  `0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
- Saved generated configuration SHA-256 (JSON.stringify encoding):
  `f35bfa01b6f487ca3c4b84c501a40148f6ecaf2b6ff31d6305ad7beb83b6c7bc`.

All nine vertex and nine edge mappings were entered through visible input
fields. A read-only comparison of the saved form exactly matches
[the corrected fixture](fixtures/postgresql-p1-mappings.json), and the generated
configuration passes `assertP1Projection`. Every vertex explicitly includes
`source_key` and `external_id`; every edge explicitly includes `source_key` and
`relationship_id`, alongside the original typed properties and endpoints.
No password was entered or reset. No source read or migration has started.

## Current approval boundary

The installed GUI displayed its native approval for new account
`af22f11b89e9434d56967573` and **Storage Blob Data Contributor for the signed-in
Azure user on that new account only**. The proposed authenticated HTTPS
endpoint is network-public; anonymous access and shared keys are disabled.
The source remains private. Standard LRS storage/request/egress charges apply.
The user approved this operation. The next observation already found the GUI
state submitted, so no duplicate submission was attempted. Azure deployment
completion is `2026-09-14T00:43:53.922411Z`; the scoped user role and private
container both succeeded. GUI read-only reconciliation now reports
`ready — public network: Disabled (provisioning is not transfer readiness)`.

Effective settings are HTTPS-only, TLS 1.2 minimum, anonymous blob access false,
shared-key access false and public network access Disabled. The role readback
confirms Storage Blob Data Contributor for the intended user at this new
account scope only. Successful policy-modify events occurred during creation
(00:43:18–00:43:24Z); the deployed template requested Enabled but the actual
account is Disabled. Do not treat provisioning success as upload readiness.
No upload or access-control override was attempted.

The next decision was whether to apply the previously used organizational
`SecurityControl=Ignore` tag **to this new trial storage account only** and
enable its authenticated HTTPS public network. This changes policy treatment
and network isolation and therefore awaits separate explicit approval. Do not
change the resource group, source firewall, anonymous/shared-key controls or
other resources. After approval, inspect the resulting settings and policy
activity and verify authenticated data access before continuing.

The user subsequently approved that exact account exception and network change.
At 2026-09-14 01:19 UTC the tag was merged without replacing ownership tags,
and public network access was enabled on this account only. Readback confirms
`SecurityControl=Ignore`, Enabled networking, HTTPS-only, TLS 1.2 minimum,
anonymous access false and shared-key access false. Authenticated container
listing succeeded; the GUI reconciled the Enabled setting. No source firewall,
resource-group tag, credential, or other storage account was changed.

The installed GUI selected the same frozen `2fd3aa4c157f` Linux archive used by
the original attempt. Local byte hashing matched SHA-256
`df8b6244963bd059389b3057274392c64164118dad0ca5949e69b04606cfa8fb`.
The GUI uploaded the 37,040,125-byte archive through the authenticated data path
and marked `developmentUpload.phase=ready`. An independent authenticated blob
properties read confirms that byte length. No new Linux execution occurred.

## Private runner approval and readiness

The GUI reconnected to this new draft and completed its live placement,
quota/SKU and ARM change preview. Preview hash:
`c2b8977ab1a788a0259eb47647eb70ae1daeed704cb37c288c2434f18facf5d4`.
Its expiry is `2026-09-14T01:39:33.799Z`; if expired when approval arrives,
refresh the preview rather than bypassing freshness admission.

Proposed VM `af-22f11b89e9434d569675` uses Japan East / zone 1, B2s_v2 at
USD 0.109/hour compute, with disk/network/NAT charged separately. Its NIC and
NSG are new; it uses the existing private compute subnet, without public IP,
SSH ingress, peering or source firewall changes. The reviewed development
artifact is installed only on this isolated new VM. The VM's managed identity
receives Storage Blob Data Reader at **this workflow's container scope only**,
not at the storage account, resource-group or subscription scope.

The user approved the pinned build and exact container-scoped identity grant.
The next observation already found the deployment submitted, so it was not
submitted again. ARM provisioning completed at `2026-09-14T01:26:26.5338161Z`.
The installed GUI reconciled the deployment and guest readiness. Readiness
operation `378f0ae6-1dfc-4668-841e-4da1a0dc58cc` confirms the exact version,
commit and archive SHA above, all expected connector capabilities, idle guest,
3.48% storage use, zero swap and zero OOM events. Boot ID is
`a7042ce1-667c-4ec4-9825-d3612e1455f7`. A subsequent GUI health refresh is
timestamped `2026-09-14T01:33:53.756Z`.

The user also directed continuation without repeating previously covered
approvals; materially different authority still requires a new decision.
The retained source VM and its existing restart-disabled PostgreSQL container
were started explicitly for this corrective attempt. Managed health command
`af-pgvm-r2-health-20260914` finished Succeeded / exit 0 at the 01:36 UTC
observation. Current certificate chain, hostname and 96-hour validity checks
pass; the certificate expires September 20 12:59:55 UTC. The retained fixture
manifest/count evidence describes 18 tables and 5,600,000 rows (not a new live
inventory). The container is running; disk use is 9%, swap and boot OOM events
are zero. No source data was recreated or changed. The new runner remains
private, with no source firewall changes.

The GUI re-reviewed the corrected source configuration and accepted the
previously covered complete-inventory read confirmation. It is now displaying
the secure **Read-only source password** input. No password was copied to chat,
no secret was persisted to the form and no inventory was submitted yet. User
credential entry is required; this is not a repeated approval request.

## Credential-wait readiness correction

After the user entered the password, the dispatch admission rejected the stale
five-minute readiness timestamp (`01:33:53.756Z`). The GUI displayed “Verify
fresh guest readiness and review source configuration first”; no assessment
intent or worker was created and the credential was discarded. This was an
interaction-timing defect, not PostgreSQL authentication failure.

The extension now rechecks idle health after interactive credential entry.
If existing health is older than four minutes, it submits one source-free
readiness command and performs bounded GET reconciliation, then applies the
unchanged five-minute dispatch gate. A changed boot, installation mismatch,
80% disk use, swap, OOM, busy worker, uncertain command or closed panel prevents
source dispatch. It never extends an old timestamp, persists credentials,
automatically retries a source read or bypasses a pending operation.

Type checking, all 177 unit tests and packaging passed, including delayed
input, unhealthy/changed boot, pending timeout and panel cancellation cases.
The VSIX was installed and VS Code reloaded while no source operation was
active. The current installed VS Code reports **1.137.0 arm64** (not the earlier
1.136.1 observation). Built and installed extension bundle SHA-256 both equal
`d024e319200752f485113d33f722936afe2aafe91df24760c51d4dc6eb85ff3b`.
The installed GUI reconnected to the same r2 workflow, displayed the revised
readiness guidance, and accepted the existing read approval. Its secure
password field is open again; end-to-end delayed-entry qualification remains
pending the new input and actual complete inventory result.

The renewed USD 800 ceiling and `2026-09-16T07:14:35.311Z` deadline are unchanged.
The read-only Cost Management refresh returned HTTP 429 again; no fresh actual
total is claimed. Existing seven VMs are deallocated and five Flexible Servers
Stopped. No resource-group locks, failed activity events or policy-modify
events were returned for the checked interval starting September 13 23:00 UTC.

Next, validate source health and perform a new complete inventory for the
changed projection through the installed GUI. Continue
through reviewed target deployment, same-VM resize, create-only migration,
strict counts and independent full canonical verification. None of these
later stages is complete; coverage remains 3/9.
