# IP-only Neo4j qualification preparation

Status: reviewed preparation, not GUI qualification. Coverage remains 7/9.

## Scope and gates

Use the existing dedicated P1 subscription/resource group. Renewed deadline is
2026-09-16T07:14:35.311Z, ceiling USD 800, conservative accrued/noncompute
reserve USD 400. Do not reset the window. Stop source/runner/target compute after
each route. Preserve all original sources, failed runs and qualification evidence.

## OP-N44 isolated source

The original `af-n44-source` is deallocated; its 128-GiB Linux Gen2 OS disk
contains the qualified P1 fixture. Create a cold disk copy and attach it to a
new specialized VM `af-op-n44-source`, D8s_v5, Japan East zone 1, without a
managed identity, public IP or new role assignment. Reuse the existing restricted
Neo4j source subnet, with verified available private IP `10.246.3.5`.
The subnet permits Bolt only from the runner subnet and denies other inbound
traffic. Original disk, VM, certificate and authentication remain untouched.

The original ARM model has no custom data, user data or VM extensions. A copied
Docker service may start, so the clone is fixture preparation only until inspected.
The copied account database is retained; do not extract its password or auth store.
Retain the cloned original container/certificate, stop it, and create an isolated
container against the copied P1 data with a new locally generated IP-SAN TLS
certificate. Export only the public CA. Retain private keys in root-owned guest
storage. Verify TLS against the literal IP, not insecure `+ssc`, and keep the
database read-only. Check free space, swap/OOM, image version and active services.

At 2026-09-15T01:16Z the source was stopped, the IP was available, no RG locks
were present, and no unexplained successful governance operation appeared in
the recent activity log. DSv5 quota was 68/100, regional total 70/101.
Azure retail Linux D8s_v5 is USD 0.496/hour. With the reviewed target/runner
USD 0.736/hour, a serial route costs USD 1.232/hour compute; even 30 hours is
USD 36.96 plus the USD 400 reserve and retained-resource charges. This is a
conservative planning estimate, not a fresh billing total (billing is delayed).

## Review and acceptance

Cold-copy avoids live database copying and mutation of a previously qualified
source. A specialized clone can retain hostname/host keys/local logs: it has no
public ingress or identity grant, and is not a production deployment template.
Inspect it before assessment; do not generalize the original disk or replay the
old bootstrap (which contains destructive container initialization).

Fixture-harness ARM operations are separate from the installed extension path.
The customer workflow must receive only IP, port, public CA and privately entered
credentials. It must not discover source resources, NICs or DNS through Azure.
Use GUI assessment, target review/deploy, same-runner resize, migration, strict
complete-count verification and the unchanged full 64-range canonical digest.
Source readiness, counts, CLI success or ARM success alone do not qualify a route.

OP-N526 follows the same isolation design only after its source/disk/network and
budget gates are independently checked. No second source is started speculatively.

## Execution / handoff

The cold copy and private specialized VM were created successfully. Original
`af-n44-source` remained deallocated. The reviewed preparation script passed
`bash -n` and ShellCheck. Guest checks showed 6% disk use, zero swap, no container
OOM and no unexpected application service. The copied old container is stopped
and retained; `af-op-n44-neo4j44` uses the copied database with TLS required and
read-only configuration. Both certificate-chain/IP verification and the actual
Bolt TLS handshake passed. Authenticated GUI assessment is still required.

Public CA SHA-256:
`91354fb6eb52e3c338b7c4333d6b8c325c9bcc9800c511ef04137cb344d901cd`.
The local public bundle is retained privately under
`production-simulation/work/op-n44-20260915/ca.crt`; no private key left the guest.
Server certificate SAN is `10.246.3.5`, valid September 15–22.

Installed GUI workflow `75e4e508-4f38-467b-b3c1-07ed05607603` selects Neo4j /
on-premises and a Japan East zone 1 B2s_v2 runner. Source resource discovery was
not used. The form has `op-n44-p1-r1`, namespace `p1`, host `10.246.3.5`, port
7687, database/user `neo4j`, stable vertex/edge property `source_key`; these fields
still need review/save with the CA selected. The release preview correctly stops
because public 2.4.0 release/checksums do not exist; reuse the separately reviewed
pinned development artifact through the GUI test-artifact flow, not a fake release.

The user approved transfer account `af75e4e5084f38467bb3c107` and **Storage Blob
Data Contributor on that account only**. The installed GUI submitted creation;
ARM and the GUI now agree that provisioning is complete. Independently verified
assignment `c34d6722-62a1-4842-96aa-2ccfd9840f1f` has only that account scope.
The public CA was selected through the GUI and source settings reviewed/saved.
Its 1,517 bytes and SHA-256 match the guest export exactly.

At approximately 01:27Z the account has public networking **Disabled**, although
the submitted template requested Enabled. Anonymous/shared-key access remain
false and minimum TLS is 1.2. The initial activity query returned no events, so
this observation alone does not identify the actor/policy. The GUI explicitly
reports that provisioning is not transfer readiness. No upload or source read
was attempted and no security-exception tag was applied. The next action needs
confirmation of the trial-account-only `SecurityControl=Ignore` exception and
authenticated HTTPS enablement, with the unchanged September 16 trial deadline.
This must not apply to the resource group or any source database.

At 01:35–01:36Z, following explicit user approval, `SecurityControl=Ignore` and
the unchanged deadline tag were merged into **this account only**, and HTTPS
public networking was enabled. ARM and GUI agree on Enabled; anonymous/shared
keys remain false, HTTPS-only true, TLS 1.2. An unauthenticated HTTPS request
reached Blob service and was rejected with `409 Public access is not permitted`.
This demonstrates network reachability and anonymous denial, not authenticated
upload success. The expiry tag is metadata, not automatic revocation.
No resource-group or source exception was applied.

The GUI selected the existing fixed development manifest, independently rehashed
locally: `2.4.0-dev.8a23a5109798`, commit
`8a23a5109798ec906109532e4cc6c32308b3c824`, 37,079,079 bytes, SHA-256
`52e1d147a13b86a729f5a993e9e72848dd87a89d0ae50a61f26459f5632444f3`.
Current dialog: **Approve pinned test artifact** for this workflow's container.
No artifact upload or VM installation has been submitted yet; this unpublished
executable requires the action-time confirmation shown by the GUI.

At 01:38Z the user approved the fixed test artifact. The installed GUI completed
authenticated upload and reports the archive prepared; durable developmentUpload
phase is `ready` with the same hash, version and byte count. The account exception
and disabled anonymous/shared-key settings were rechecked unchanged.

The GUI reconnected to this draft and refreshed the VM preview at
`2026-09-15T01:39:54.398Z`. B2s_v2 / Japan East zone 1 is USD 0.109/hour compute.
The preview has no public IP or source firewall change. Its sole role assignment
is Blob Reader on this workflow container for the new VM identity (not the RG,
subscription or source database). Current dialog: **Create reviewed runner** for
`af-75e4e5084f38467bb3c1`. This creation/grant has not been submitted. Revalidate
preview expiry before confirming if the user responds later.

The source clone is confirmed deallocated; no runner or target for this route
has been deployed. After resolving transfer readiness, select the fixed artifact
through the GUI and later restart only this clone before authenticated assessment.
All prior evidence remains retained. Qualification coverage remains 7/9.

## 01:43–01:47Z — runner provisioned; protected password handoff

The user approved this runner and continued routine actions for the route.
The installed GUI submitted once at `2026-09-15T01:43:19.319Z`; deployment
succeeded, then GUI reconciliation marked it provisioned at `01:44:10.356Z`.
Identity `6b46cc50-edef-4a31-9454-0bdee640ce46` has independently verified Blob
Reader only on the workflow container. The original sources remain stopped;
only the isolated OP-N44 clone was restarted for this assessment.

Initial ready operation `a7dd16c8-92c5-4aea-8f77-b938d0af0971` failed because
installation was still finishing; its ARM command/evidence are retained.
Cloud-init finished at `01:44:39Z`. Explicit fresh ready operation
`2d58ed1c-da5e-4385-b0a1-d9a0555ded29` passed at `01:45:59.825Z`:
boot `090640bd-8084-49b8-990c-605dbc085933`, matching pinned build/hash and
Neo4j inventory/migration capabilities, idle, 3.508270134% disk, zero swap/OOM.
Source startup also initially refused connections; at `01:45:27Z` logs confirmed
4.4.48 started and the actual literal-IP TLS handshake passed. Neither early
startup observation was accepted as readiness. The copied old container is still
stopped, the isolated TLS container running with zero restarts/OOM.

The installed GUI reopened/reviewed the saved IP-only source form and approved
complete source inventory. Its protected **Read-only source password** input is
waiting for the user to enter the existing AZ-N44 `neo4j` credential and Enter.
No password was extracted from retained files or the copied auth database.
No source inventory has been submitted before that entry. The user was asked
only for private credential input, not another routine workflow approval.
Runner and source are running (USD 0.109 + 0.496/hour compute); target is not
created. Budget USD 800 / reserve USD 400 / September 16 deadline unchanged.

## 01:54Z — first authenticated inventory rejected

Following user password entry, the GUI refreshed idle guest readiness at
`01:54:04.014Z`, then submitted complete inventory operation
`4864b359-4e9b-477e-9c4e-163da4462f34`. It failed at
`01:54:23.971296230Z` (exit 1). The operation identity and configuration hashes
remain retained; no report was accepted and no target was created.

A read-only guest diagnostic classified the 89-byte private stderr as Neo4j
`Security.Unauthorized`. Its SHA-256 is
`05a83f01f9f6a512030e8b3f8daac11c4a7ff44974bf62f1fec9df7d8fdadbe7`.
Only the category, size and checksum were returned, not raw stderr or secrets.
Runner disk was 4% (rounded), swap zero. This indicates rejected source
credentials, not TLS/network failure or a successful assessment. Correct the
existing credential privately; do not extract authentication files or change the
source password. Retain the failure and explicitly prepare a fresh GUI attempt
after refreshed idle readiness. Routine route approvals remain authorized.
