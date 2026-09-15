# AZ-COSMOS r2: typed mapping installed-GUI qualification

Updated: 2026-09-15T01:08Z. Outcome: **PASS**.
Overall GUI coverage is **7/9**. Old r1 graph/jobs/evidence are unchanged.

## Final full canonical qualification PASS

The installed GUI displays **P1 full canonical digest: PASS**. Report generated
at `2026-09-15T01:01:19.174440026Z`, operation
`cde9e6fe-ab51-45e6-8427-c978a748321e`, job
`d5edef98-bb51-4040-b6ed-0274e252de26`. All 1,600,000 vertices and 4,000,000
edges agree across all 64 ranges, including typed properties, stable identities
and endpoints. Expected and actual canonical root:
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.

GUI export/import passed. Report: 23,220 bytes, SHA-256
`a5d14d4b2f18673ac2fe12d59d46e195f14f174ff4f435ef8deba427dda9efe9`.
An independent local check validated report bytes/hash, exact job/read-only
identity, all 64 expected/actual leaves and 5.6M-row totals, and recomputed
both canonical roots from every leaf. No alternate root, target patch,
relaxed numeric comparison or failed-job replay was used. This qualifies the
installed-GUI AZ-COSMOS route with explicit `float64` mappings; it does not
claim production-scale coverage or qualify either remaining on-premises Neo4j
route. Compute stop/deallocation is complete; all data is retained.

Post-verification GUI health at `2026-09-15T01:04:22.920Z`: idle, disk 5.098%,
swap/OOM zero, same boot and loader. After confirming resource ownership,
only this route's runner and target received deallocate/stop requests.
By `2026-09-15T01:08Z`, all 13 trial VMs were confirmed deallocated and all
11 Flexible Servers Stopped. No data, disks, roles or evidence were deleted.
Cosmos and retained storage can still incur charges; stopping compute is not
zero billing. Flexible Server automatically restarts after seven days unless
managed separately. The existing September 16 deadline is unchanged.

## Earlier phase: migration/counts PASS; independent verifier execution

The user explicitly approved the separate verifier. Stale guest health was
refreshed through the GUI at `2026-09-15T00:56:11.184Z`: idle, disk 3.530%,
swap/OOM zero, same boot and pinned loader. No group locks; returned governance
events were policy audits, not control changes. Target storage is 14.043%.
The unchanged verifier manifest and exact job were reviewed again, then the
installed GUI submitted operation `cde9e6fe-ab51-45e6-8427-c978a748321e` once
at `2026-09-15T00:58:29.341Z`. Full canonical results are still pending.

Fresh job `d5edef98-bb51-4040-b6ed-0274e252de26` finished. The counts report
generated at `2026-09-15T00:36:07.826405692Z` passes all 24 checks and all
18 label counts: 1,600,000 vertices and 4,000,000 edges, zero rejects,
no failed/incomplete checks or errors. Submission-to-report: about 13m46s.
The installed GUI transferred and imported 9,619 bytes, displaying
**Exact source and target counts agree with no rejects**. Independent local
SHA-256 agrees:
`f1824243ff13d1cc2f44493c47151a68d30be5d92b353dc87347b5326624c938`.
Configuration fingerprint:
`a72c1dc145815a22fea5e275cafaab4435d20a60995a99884d9343885199e37b`.

A read-only guest diagnostic found no running loader, retained load/verify/report
files, about 4% filesystem usage, no swap or kernel OOM messages. Post-load
installed-GUI health at `2026-09-15T00:42:48.219Z`: idle true, disk 3.529%,
swap/OOM zero, matching binary and post-resize boot. Target storage was below
13% during the observed load (80% gate unchanged). No failed run was replayed.

The GUI reviewed **Approve full P1 verification** for the independently built
`2.4.0-dev.8a23a5109798` verifier, SHA-256
`60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d`.
Its manifest was selected from `work/vscode-p1-verifier.YxRRzz/manifest.json`;
execution was submitted only after the user's action-time approval.
It reads this existing target only, regenerates the frozen fixture and compares
all 5.6M records / 64 ranges; no graph, loader, credential or network changes.
Bounds: 4 GiB RAM, about 1 GiB retained fixture, 25 minutes; private report
return through existing storage.
Both new compute resources remain running for verification; USD 800 /
USD 400 reserve / `2026-09-16T07:14:35.311Z` remain unchanged.
Full typed canonical qualification is still pending, so coverage remains 6/9.

## Earlier fresh typed migration submission

Target provisioning completed without repair or replay. GUI AGE restart,
submitted `2026-09-15T00:14:40.405Z`, is finished; live preload value is
`pg_stat_statements,age`, restart pending false. Same-VM resize began at
`00:16:28.250Z` and finished at D4s_v5. Preservation hash:
`44e0e188183d68a1553fa70906deffef1dc40adfeb18e74bb7dd88d1356c3fab`.
Post-resize GUI health at `00:21:19.630Z`: boot
`ce0398d0-419f-459f-ab8f-d25708e22899`, matching pinned artifact/capability,
idle, disk 3.482%, swap/OOM zero. The new target is private and Ready.
Policy-created NSG/advanced-threat-protection changes were observed and
preserved; the delegated subnet NSG has no custom rules. No group locks.

The installed GUI approved and submitted fresh create-only migration job
`d5edef98-bb51-4040-b6ed-0274e252de26` at `2026-09-15T00:22:21.554Z`.
The same UUID identifies its retained operation. Typed configuration and
complete inventory are unchanged. No source credentials were requested:
Cosmos uses the approved managed identity. The GUI will migrate, then run
strict complete counts verification. Neither counts nor independent canonical
verification has yet passed; do not mark the route qualified.

## Completed

- Mac unlocked; VS Code window reloaded. Updated installed GUI visibly exposes
  Cosmos `name=field:type` instructions and typed mapping fields.
- Fresh GUI workflow: `d138f4e4-bcf3-40fe-a876-ee9ce062e08a`. Source selection
  used subscription, resource group, Discover, and `afcosmosp120260907`.
  Placement: Japan East / zone 1, existing private runner subnet, initial B2s_v2.
  Cosmos data region, not a logical source availability zone, is checked.
- Source name `az-cosmos-p1-r2`, namespace/database `p1`: all nine vertex and
  nine edge mappings entered and reviewed in the GUI. The persisted form is
  independently exactly equal to `fixtures/cosmos-p1-typed-mappings.json`.
  All `score` and `distance_km` properties explicitly declare `float64`.
- No Azure write, source read, deployment or migration submitted this turn.
  Workflow remains `draft`; no credential entered or extracted.

## Live read-only gates

- All 12 existing VMs deallocated; all 10 Flexible Servers Stopped.
- Cosmos public network Disabled, local/key authentication disabled, private
  endpoint Approved, actual data region Japan East; no RG locks returned.
- External activity includes storage, Defender and Event Grid writes around
  19:33–19:44Z by another principal, plus policy audit results. No controls were
  reverted. The old r1 transfer account still has authenticated public HTTPS,
  no anonymous access, shared keys disabled and TLS1.2.
- Cost Management returned 429 once; not repeatedly retried. Last available
  cost is delayed, not final. USD 800 ceiling, USD 400 reserve and
  `2026-09-16T07:14:35.311Z` deadline unchanged; about 32h57m remain.
  Reviewed combined runner/target compute USD 0.736/hour would add about
  USD 24.25 through the deadline. This is an estimate, not current billing;
  retained Cosmos/storage charges remain separate.

## Cosmos grant and complete typed inventory passed

The user approved the new runner's Cosmos read-only grant. The installed GUI
submitted assignment `0caf9405-337c-4b6f-879c-b40900c9c03b` once at
`2026-09-14T23:47:50.199Z`, then reconciled it to ready. Independent ARM lookup
confirms principal `0132a1ea-492d-41d3-b910-b5d978a5a06c`, Built-in Data Reader
and scope exactly the test source account. No keys, writes or public access
were added to Cosmos.

After GUI mapping review and the normal source-read confirmation, complete
inventory `5aabd41f-38bb-4b24-8623-27ae92a0ccc0` was submitted at
`2026-09-14T23:48:31.379Z`. Fresh same-boot health passed before submission.
GUI status advanced from accepted to running; no replay. Declared configuration
hash `ce431d5d750b17a4b56d6a3c8f28f3b4cdc387cffaa90a4ecb8e3eff8ac40a14`,
guest configuration hash
`7f6908039e53e5e54eb75160f0733a5bb7a78611bd650648155e835f0c79b974`.
Limits remain 30 minutes / 4 GiB / no swap; frozen source must remain unchanged.
The complete inventory finished at `2026-09-14T23:57:35.722167732Z`: PASS,
all 18 mappings reached EOF, 1,600,000 vertices plus 4,000,000 edges,
no errors or incomplete checks. The GUI exported/imported the 2,940-byte report;
independent SHA-256 agrees:
`8327409449b181afca0b205a7760bf0f766b4b197ba08dd8c88fe57b8f7bf17e`.
Mapped record bytes: 458,398,000; sizing high bound: 8,567,972,000 bytes.
Inventory is not migration or canonical qualification.

The first target preflight correctly rejected stale guest readiness without
submitting a target. GUI readiness was refreshed at `2026-09-15T00:05:39.093Z`:
same boot and pinned binary, idle true, disk 3.481%, swap/OOM zero.
Fresh GUI target review passed: private PostgreSQL 18 / AGE, D4ds_v5,
128 GiB, Japan East zone 1, new non-overlapping `10.246.15.0/24` subnet,
same-runner D4s_v5 resize later. Combined compute USD 0.736/hour,
USD 400 reserve, USD 800 ceiling and original September 16 deadline.
Activity review showed expected test Run Commands and policy audits; no locks
were returned. No controls were reverted. The GUI saved the reviewed LoadJob
and target plan in the existing local Cosmos staging folder, without replacing
r1 files. Plan hash:
`d5c98c8baaf9e442d4ba8ac30e0aee73682a70e66d4182936913e3b25c68a7fe`.
Target deployment was submitted once through the GUI. Independent ARM status
is Running at `2026-09-15T00:08:02.748245Z`, without a reported error; GUI
reconciliation shows submitted and does not replay it. Provisioning is the
active step; no migration or canonical verification has begun.

## Earlier private runner readiness and grant handoff (completed)

The user approved the private VM and container-scoped Blob Reader grant.
The unchanged fresh preview was confirmed and submitted once through the
installed GUI at `2026-09-14T23:43:47.815Z`. ARM and GUI report provisioned.
VM `af-d138f4e4bcf340fea876` is running, B2s_v2, zone 1, no public IP.
System-assigned principal: `0132a1ea-492d-41d3-b910-b5d978a5a06c`.
The retained role assignment matches Storage Blob Data Reader at the approved
transfer container only; no source permission was included.

GUI Linux readiness operation `d6abe3de-a37c-452d-bba0-b90676b04206` finished
and was reconciled at the GUI. Retained check timestamp:
`2026-09-14T23:45:06.082Z`. Version `2.4.0-dev.8a23a5109798`, implementation
commit and archive SHA-256 match the reviewed pinned artifact; the guest
advertises `cosmos-explicit-property-types-v1`. Boot ID
`71e307e5-3f09-4b84-b989-2a1c4573f824`; idle true, disk 3.4784649594897576%,
swap 0, OOM events 0. No source reads or migration have started.

**Earlier GUI handoff (now completed):** the new runner's **Grant Data Reader** confirmation
for `afcosmosp120260907` only is open and has not been accepted. The proposed
Cosmos DB Built-in Data Reader assignment permits reads, not writes, does not
enable public networking or keys, and does not start assessment. The new
principal's access requires action-time approval. Source remains private.
Refresh Linux readiness if stale before the separately approved inventory.
The new VM incurs USD 0.109/hour compute while awaiting input; old compute
was not restarted. Budget/expiry are unchanged.

## Earlier storage exception, upload and runner preview (completed)

The user explicitly approved the new account's network exception. At about
`2026-09-14T23:37Z`, only `afd138f4e4bcf340fea876ee` received merged tags
`SecurityControl=Ignore` and `expiresAt=2026-09-16T07:14:35.311Z`, followed by
Public Network Access Enabled. Anonymous access and shared keys remain disabled;
TLS1.2 and existing ownership tags are unchanged. The expiry tag documents the
authorized deadline; it is not proof of automatic revocation. An authenticated
container read succeeded. Cosmos remains public Disabled / local keys disabled.

The installed GUI selected and uploaded the reviewed development manifest from
`work/vscode-runner-build.GXKmVv/manifest.json`, then reported the pinned archive
prepared. Persisted upload is `ready`. Independent blob properties confirm
37,079,079 bytes and SHA-256 metadata
`52e1d147a13b86a729f5a993e9e72848dd87a89d0ae50a61f26459f5632444f3`.
The uploader also validated local bytes against the pinned manifest. No new
binary was built or substituted.

After reconnecting the draft through the GUI, fresh VM preview passed at
`2026-09-14T23:40:24.565Z`; hash
`f310f187c0e8d3d0ffffa887586711562b3a562040d31da7a387bcc6f4423d64`.
Preview expires at `23:55:24.565Z` and must be refreshed if stale. Proposed VM
`af-d138f4e4bcf340fea876` is B2s_v2, Japan East zone 1, in the existing private
runner subnet, USD 0.109/hour compute plus storage/network. No public IP,
SSH ingress, peering or source firewall change is included.

**Earlier handoff (now completed):** approve this fresh private VM and its managed identity's
Storage Blob Data Reader grant scoped only to
`af-d138f4e4-bcf3-40fe-a876-ee9ce062e08a` in the new transfer account.
It installs/runs the reviewed unpublished `2.4.0-dev.8a23a5109798` artifact.
No VM deployment has been submitted; state is `previewed`. Cosmos Data Reader
is not part of this preview and remains a separate later access decision.
No source inventory, migration or canonical verification has started.

## Earlier storage approval and connectivity gate (resolved)

At the user's explicit action-time approval, **Create storage and scoped role**
was pressed at approximately `2026-09-14T22:48:47Z`. Deployment succeeded;
the GUI reconciled storage to ready. Independent ARM reads confirmed the
account-scoped role assignment `db222a8c-5b6d-4cb6-b4af-8740911f87e6` grants
the signed-in user Storage Blob Data Contributor only on this new account.

However, live Public Network Access is **Disabled**, despite the reviewed
template requesting Enabled. The account activity log includes successful
`Microsoft.Authorization/policies/modify/action` at
`2026-09-14T22:48:58.7262832Z`; this supports policy modification during creation.
The GUI explicitly reports `ready — public network: Disabled (provisioning is
not transfer readiness)`. Shared keys and anonymous access remain disabled,
TLS1.2 remains configured. No upload, VM deployment, assessment or migration
has begun. Do not equate provisioning success with transfer readiness.

The current handoff is whether to apply the same trial-storage-only official
`SecurityControl=Ignore` exception plus unchanged expiry and authenticated
public HTTPS used for r1, now to `afd138f4e4bcf340fea876ee` only. No exception,
network re-enablement or policy change has been applied to this new account.
Source networks and authentication must remain unchanged.

### Earlier approval handoff (resolved)

GUI shows **Create dedicated transfer storage and grant your Azure user data
access?** for `afd138f4e4bcf340fea876ee` in `rg-af-vscode-p1-20260905-a`.
It would grant the signed-in user **Storage Blob Data Contributor on this NEW
account only**. Standard LRS/request/egress charges apply; HTTPS is network-public,
anonymous access and shared keys disabled. No source server is exposed.
The final **Create storage and scoped role** button has **not** been pressed.
New security-sensitive access requires action-time confirmation.

## Next

AZ-COSMOS r2 is qualified; do not replay it. Retain all r1/r2 records and data.
The two remaining GUI routes are OP-N44 and OP-N526: IP/port-only simulated
on-premises Neo4j, without source discovery through Azure APIs. They require
their own GUI source configuration, migration and complete canonical checks.
The existing budget and deadline remain binding; do not reset them by route.
