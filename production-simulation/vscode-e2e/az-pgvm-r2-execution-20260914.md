# AZ-PGVM corrective GUI attempt

Status: corrected source draft reviewed; authenticated artifact transfer and
private Linux runner readiness and source health passed. The corrected GUI
submitted complete inventory with the saved random credential. The fresh
inventory passed all 5,600,000 rows and was hash-verified in the installed GUI.
The earlier authentication failure remains retained. The new private target
deployment, AGE preload restart and same-VM resize are complete. Post-boot
readiness passed; the GUI is at migration credential entry.
No migration qualification is claimed.

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

## First corrected-projection inventory: authentication failure

On the next password entry, the installed GUI automatically refreshed Linux
readiness at `2026-09-14T02:05:15.879Z`, preserving the same boot, pinned build,
idle worker, 3.48% disk use and zero swap/OOM. It then submitted exactly one
inventory at `02:05:29.119Z`: operation
`b78461db-10aa-482c-a75f-ae22d551fe0d`. The saved configuration remains
`f35bfa01b6f487ca3c4b84c501a40148f6ecaf2b6ff31d6305ad7beb83b6c7bc`;
guest configuration is
`9750d398f27fe1f9d591a5b128d2dc8bd9863b4d43c9adbb5ed64fa972d2ad71`.
This confirms the delayed-entry fix reached protected source dispatch rather
than being rejected by stale readiness.

The worker ran from `02:05:37.537282585Z` to `02:05:37.695986033Z`, exit 1.
Its 51-byte initialization-error log has SHA-256
`d1d694a1717436e62073c3f131686fba9f5f11455eb128dd409544982872d94b`.
GUI GET reconciliation marked the assessment failed, without replay. The
source container logs for 02:05–02:07 UTC contain one password-authentication
failure; the container remains running with OOM false. The runner resolves the
reviewed hostname to the expected private source IP. The generic CLI error
alone did not establish authentication failure; the server log supplies that
additional evidence. No password, connection string or raw private diagnostics
were printed. No password reset was performed.

Read-only diagnostic command evidence is retained, including one diagnostic
that could not run because the minimal runner has no jq, and a TLS diagnostic
that found the transient CA file already removed by the worker's normal
cleanup. Neither diagnostic failure is an assessment result or proof of bad
TLS. No source/target data or operation directories were deleted.

The next step is fresh same-boot idle health, explicit retention of this failed
assessment, re-review of unchanged mappings, and input of the correct existing
`agefreighter_reader` password. A fresh attempt must have a new operation ID;
this failed operation must never be resumed. The Cost Management query again
returned 429; the USD 800 ceiling / September 16 deadline are unchanged. No
resource-group locks or failed/policy-modify governance events were returned
for the pre-assessment review interval. Qualification remains 3/9.

Subsequently the GUI refreshed same-boot idle readiness at `02:11:42.480Z`.
While the native failure-retention confirmation was open, the user acted in
VS Code. Readback confirms the failed operation is retained in
`assessmentHistory`, with no current assessment and no replay. The GUI
re-reviewed the unchanged source and accepted the previously covered read
approval, then opened its secure password field. It is waiting for corrected
existing credentials; there had been no second inventory submission at that
checkpoint.

## Saved credential provenance and fresh inventory

The local test password file is owner-only readable/writable and contains a
64-character hexadecimal string. The creation record at
`2026-09-06T23:26:23.288Z` uses `openssl rand -hex 32`; subsequent preparation
calls pass the saved value as `sourcePassword`. It is a randomly generated
password, not a password hash to decode or a PostgreSQL SCRAM/MD5 verifier.
The value was not printed in chat or in these results.

After the user entered that saved value, installed-GUI readiness refreshed at
`2026-09-14T02:17:49.661Z` and a fresh inventory was submitted at
`02:18:15.164Z`, operation `5132c60a-fddb-47d3-89f0-4e189718f6cf`.
The approved configuration SHA remains unchanged. Guest configuration SHA is
`16dc629fc075ae6e9a5c2949696fed4d751e5ff2b7770a3752be6d517b34ee8e`.
GUI status progressed from accepted through running to finished; the earlier
failed operation remains in history and was not resumed. The PASS report was
generated at `2026-09-14T02:21:06.029971691Z`: 1,600,000 vertices, 4,000,000
edges, all 18 exact mapping counts, zero errors and zero incomplete checks,
using one complete repeatable-read snapshot. It reports 458,398,000 decoded
mapped-record bytes and a storage sizing range of
3,008,790,000–8,567,972,000 bytes before target headroom. The installed GUI
exported and imported the exact 2,944-byte report and displayed it as
hash-verified. SHA-256:
`33c83a3021d9691333a0220f678e3bbf93d92533fc968b38fb5eb5992dfc5c0e`.
This confirms the saved plaintext random password is accepted by the actual
source. It is an inventory success, not a target migration or canonical digest
qualification. The reviewed private VNet has unused `10.246.10.0/24` space for
the separate r2 target; previous target subnets and graphs remain untouched.

## Private target deployment submitted

After complete-inventory import and another idle readiness check
(`02:26:57.685Z`), the installed GUI reviewed current placement, quota and
retail-price evidence. It saved the secret-reference-only LoadJob and plan in
the local trial staging folder and submitted a new private target deployment.
ARM reports Running, timestamp `2026-09-14T02:31:21.624418Z`.

- New target: `afpg-22f11b89e9434d569675`, PostgreSQL 18 / AGE.
- Same trial resource group, Japan East, zone 1; no public access or peering.
- New delegated subnet `10.246.10.0/24`, private DNS in the existing VNet.
- General Purpose `Standard_D4ds_v5`, 128 GiB; HA disabled for this trial.
- Same runner planned for `Standard_D4s_v5`; resize has not started.
- Combined target/loader compute USD 0.736/hour; accrued/non-compute reserve
  USD 400, total ceiling USD 800, deadline `2026-09-16T07:14:35.311Z`.
- Plan SHA-256:
  `b866f3826105ef8020e936486f33465e916b7c66d675325f86d50a069a8771eb`.

Existing graphs, source data and failed evidence are untouched. Only the new
runner and retained PostgreSQL source VM are running; the other six VMs remain
deallocated. The previous five Flexible Servers remain stopped and the new
one is provisioning. The read-only Cost Management refresh returned HTTP 429
again; no fresh actual billed total is claimed. No resource-group locks or
failed/policy-modify governance events were returned for the current review
interval. The last command-slot check found 15 of the new VM's 25 managed Run
Command entries occupied; preserve evidence and check capacity before later
dispatches.

Next, reconcile target deployment and AGE readiness, then same-VM resize,
create-only migration,
strict counts and independent full canonical verification. None of these
later stages is complete; coverage remains 3/9.

## Target readiness and migration-entry correction

The installed GUI reconciled target provisioning and the AGE preload restart
submitted at `2026-09-14T02:41:46.571Z`; both are finished. Independent ARM
readback confirms Ready, D4ds_v5, zone 1, the intended delegated subnet and
private DNS, and public network access Disabled. Resource-group locks are
absent. Subsequent policy audit events include two failed subnet audits and
a successful target deployIfNotExists event; these are not migration results
or grounds to bypass policy. Effective target isolation remains unchanged.

The credential-wait readiness correction now also covers migration entry.
After the private source password input, the extension refreshes stale idle
health before creating the first migration intent, preserving completed
inventory, target and resize evidence. Existing migration intents still reject
this path; credentials are not persisted and no source operation is replayed.
All 178 tests and packaging pass. The installed extension bundle matches
SHA-256 `d1fa4b61d344269baea8071939921e3de0300bd9596acd9df142e0c1dc3001ae`;
VS Code was reloaded and reconnected to the same r2 workflow.

GUI readiness at `02:49:47.486Z` confirms idle, disk 3.48%, no swap/OOM and
the same pinned Linux build. GUI same-VM resize began at `02:51:22.074Z`.
Deallocation completed and the D4s_v5 size update was submitted. The retained
disk/NIC/identity/placement fingerprint is
`e98e6d4a15cc676e9db7b6da19fb5fb1523e61f2d6318c4110498c90aa0c16c8`.
Source data and all prior failed graphs/evidence remain untouched.

GUI reconciliation subsequently confirms resize finished with the same
preservation fingerprint. Post-boot readiness at `2026-09-14T02:58:03.871Z`
confirms boot `de2d194e-efa7-40f2-b350-839b2a60ea2e`, the unchanged pinned
runtime, idle guest, storage 3.50%, swap 0 and OOM 0. Managed Run Command
capacity is 17/25 before migration. The GUI migration preflight passed and
the previously covered create-only 5,600,000-row migration/counts confirmation
was accepted. The secure source-password field is required next; no migration
job has been submitted at this checkpoint. Full property digest qualification
is still a separate remaining step, and overall coverage remains 3/9.
