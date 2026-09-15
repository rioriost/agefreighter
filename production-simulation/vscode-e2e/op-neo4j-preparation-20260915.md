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

## 02:09–02:16Z — corrected credential and complete inventory PASS

The existing AZ-N44 credential is a generic-password item in the Mac Keychain:
service `agefreighter-az-n44-neo4j`, account `agefreighter`, not a plaintext file
in the repository. The user retrieved it and privately re-entered it into VS Code.
No source credential was changed. The previous rejected operation remains in
assessment history and on the guest.

Fresh inventory `0f3d9bcf-355c-41f1-a16f-68f4cf88ad4f` passed at
`2026-09-15T02:09:00.145645164Z`, after fresh readiness at `02:08:19.770Z`.
It has exact transactional count-store totals: 1,600,000 vertices, 4,000,000 edges,
5,600,000 records. Errors, warnings and incomplete checks are empty. Installed
GUI transfer/import and independent local verification agree on 663 bytes and
SHA-256 `6373da2080e0ce7aea06404c4b6831d58e57308b27c6a8ee72409b0dd818753d`.
This is complete source inventory, not full property validation or qualification.

The GUI refreshed healthy idle readiness at `02:12:53.078Z`: matching build and
boot, disk 3.511%, no swap/OOM. All old Flexible Servers remain Stopped, and only
the OP-N44 source and runner are running. No RG locks or unexpected successful
non-audit/non-run-command actions were returned for the 02:00Z onward check.
The reviewed fresh delegated subnet `10.246.16.0/24` is non-overlapping.

At approximately 02:16Z the GUI saved secret-reference-only LoadJob and target
plan under `production-simulation/work/op-n44-20260915`. Plan SHA-256:
`250e812e6c4a3e7907e833dcf367ce6f4eab2e221af164c5fe965719f8eab25e`.
Target `afpg-75e4e5084f38467bb3c1`: PostgreSQL 18 / AGE, D4ds_v5, 128 GiB,
Japan East zone 1, private DNS/subnet, no public access, no HA for this trial.
Same runner later resizes to D4s_v5. Target+runner compute is USD 0.736/hour;
source is additional USD 0.496/hour. Ceiling USD 800, accrued/noncompute reserve
USD 400, deadline `2026-09-16T07:14:35.311Z` unchanged. No automatic shutdown
is implied by these metadata. Deployment/AGE readiness, resize, migration and
both strict counts and the frozen full canonical digest are still separate gates.

## 02:23–02:30Z — private target and same-VM resize complete

GUI reconciliation accepted target provisioning at approximately 02:23Z.
The reviewed AGE preload restart submitted at `02:23:40.333Z` subsequently
completed; ARM is Ready with `pg_stat_statements,age`, no pending restart and
public networking Disabled. A policy deployIfNotExists action and a separate
advanced-threat-protection settings deployment were observed for this new target;
no security control was disabled to proceed.

The GUI performed and reconciled each same-runner step separately: deallocate
(`02:25:34.356Z`), resize to D4s_v5, then start. Disk/NIC/system identity and
placement invariants remain unchanged; principal is still
`6b46cc50-edef-4a31-9454-0bdee640ce46`. GUI resize phase is finished.
Fresh readiness at `02:29:49.488Z` proves new boot
`34a6795d-179b-485f-96bd-03e1b0c160a9`, unchanged pinned version/hash/capabilities,
idle, disk 3.512544759%, no swap/OOM. No source VM resize or credential change.
The GUI is advancing to the separately approved migration; it requests the
existing Neo4j credential privately again because inventory does not retain it.
No migration is claimed before a durable job and its actual execution evidence.

## 02:35Z onward — first migration running

After the user entered the existing source password privately, the GUI refreshed
readiness (`02:34:10.543Z`, matching boot/build, idle, disk 3.514%, swap/OOM zero)
and submitted migration job `6ad5c2d0-9d25-4ceb-a382-9516af0c22cc` once at
`02:34:53.311Z`. Guest execution began at `02:35:24.720350577Z`; private target
preparation completed. Guest configuration SHA-256 is
`1d92b2e4e728f305bdaaa5a43461b41f18b6d85ed6157f84ff5befc75206e12a`.
The pinned fixed CLI performs load followed by `verify --counts --require-complete`.
Counts alone remain distinct from the subsequent frozen 64-range property digest.

Read-only guest observation at `02:41:14Z`: same job running, loader RSS
35,404 KiB, root disk 4% rounded, no swap or boot OOM messages. No completed
load/verify artifact existed at that observation. Available target storage metrics
through `02:36Z` showed 6.864% maximum. No mutation/replay was used to monitor.
The verified existing P1 archive retains SHA-256
`60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d`.
It has not yet been executed for this job. Both previous failures and all original
source/target evidence remain retained. Budget and deadline are unchanged.

## 02:45–02:58Z — migration and strict counts PASS; full digest confirmation pending

Guest execution finished at `02:44:54.736459959Z`, exit 0. The installed GUI
imported the final report and displays migration finished / counts pass.
Independent local hashing agrees on 9,619 bytes and SHA-256
`f80bac38b0af32a553a2393b434615f5bc526bf589c52ef7a7b2071700747785`.
The report generated at `02:44:53.578739467Z` passes all 24 checks and all
18 label counters: 1,600,000 vertices, 4,000,000 edges, zero rejected rows,
errors, warnings and incomplete checks. Durable job is unchanged;
configuration fingerprint is
`febb6f2ae5b5fd1a152bee4414bf8ca4b0fddc795c7b392f86bf934d2b99a4c5`.

No loader process remains at the `02:45:10Z` guest observation. Fresh GUI
readiness passed at `02:53:39.521Z`. Target storage through `02:53Z` peaks at
13.8023%, below the 80% gate. The activity query returned policy audit records;
no security protection was changed in this phase.

The installed GUI selected the unchanged verifier manifest and shows the final
`Approve full P1 verification` dialog for this existing job. It is deliberately
not submitted yet: execution of an unpublished development verifier requires
action-time confirmation. The dialog specifies read-only full comparison of
5.6M records / 64 ranges, unchanged pinned commit/archive, no loader/graph/
credential/network changes, up to 4 GiB RAM, approximately 1 GiB retained fixture
and a 25-minute cap, with results returned privately through existing storage.
Counts PASS is not full qualification. Compute remains running while awaiting
confirmation; original budget/deadline remain in force. After full canonical
PASS, stop the current two VMs and Flexible Server without deleting evidence.
