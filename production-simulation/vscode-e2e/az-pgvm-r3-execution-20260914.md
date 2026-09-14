# AZ-PGVM native-float corrective GUI attempt

Status: fixed development artifact installed on the new private VM; installed-GUI
readiness and source health pass. After private credential entry, a new complete
inventory has passed and its hash-verified report is imported. Private target
deployment, AGE preload restart and same-VM resize are complete. Post-boot
readiness passes; fresh migration and complete counts verification pass.
Independent full P1 canonical verification is running.

## Scope and preserved evidence

- The released v2.3.1 fix (`952b6b4`) was merged into guided development in
  `43490f6`; `9ef1696` adds the PostgreSQL native-float guest capability gate.
- Extension remains 2.4.0; runner-first flow and Azure Resources authentication
  remain. No 2.4.0 release or Marketplace publication was performed.
- AZ-PGVM r1/r2 targets, jobs, failed digests and guest evidence remain unchanged.
  Old PostgreSQL checkpoints are not resumable with the new fingerprint.
- Overall GUI qualification remains 3/9. This attempt must use a fresh target
  and job and match all 5,600,000 records / 64 canonical digest ranges.
- Renewed trial budget remains USD 800 and deadline
  `2026-09-16T07:14:35.311Z`; no P3 authorization is reused.

## Local validation and artifacts

All Go package tests passed. PostgreSQL/runner race tests passed. Live local
PostgreSQL 18 / AGE tests preserve exact native float serialization in COPY,
cursor and keyset, with pre-encoding on/off, arrays/domains, non-finite rejection
and legacy-checkpoint refusal. The first AGE test invocation used the wrong
local database role and failed authentication; rerunning with the container's
configured role passed all three modes. Temporary local test containers are
stopped again, not deleted.

Extension typecheck/build and 180 unit tests passed. The newly packaged VSIX
was installed and reloaded in the actual Mac VS Code:

- VSIX SHA-256: `95e6864210bc2ace043d2685ece28d9892b0f091dee51e9ba902dfd4f31c120c`.
- Linux version: `2.4.0-dev.9ef16968363b`.
- Commit: `9ef16968363b31214324f392553f7c8e88150272`.
- Archive SHA-256: `10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
- Archive bytes: 37,056,164.
- Local build manifest: `production-simulation/work/vscode-runner-build.2cDvou/manifest.json`.

## Fresh GUI workflow

- Workflow: `c275d043-de93-4b0a-b2b0-59cddd13c84f`.
- Name: `az-pgvm-p1-r3`.
- Azure discovery selected the retained PostgreSQL source VM in the trial group.
- All 18 mappings were entered into visible GUI fields; a read-only comparison
  of the saved form exactly matches `fixtures/postgresql-p1-mappings.json`.
- Configuration SHA-256 (JSON.stringify encoding):
  `691b33014f5aa54f0806d24269851cf7b442e32498959db4f59e71517185e426`.
- Existing public CA SHA-256:
  `0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
- Transfer account: `afc275d043de934b0ab2b059`; dedicated workflow container.
- Runner: `af-c275d043de934b0ab2b0`, Japan East / zone 1 / B2s_v2;
  compute estimate USD 0.109/hour plus disk/network.

The transfer deployment succeeded. Effective public network access was initially
Disabled; within the previously approved storage-only exception, the new test
account received `SecurityControl=Ignore` and authenticated public HTTPS access.
Anonymous access and shared keys remain disabled; HTTPS-only and TLS 1.2 remain
required. No source firewall, VNet peering or public VM IP was added. Installed
GUI upload independently verified the pinned archive and recorded `ready`.
The VM preview and its container-only Blob Reader grant were reviewed; the GUI
submitted a new deployment once. Provisioning is not guest readiness.

## Live gates at resumption

Cost Management returned USD 17.5157291900634 for September 12 and
USD 15.4061238600271 for September 13 (USD 32.9218530500905 total in returned
rows). Billing lags; this is not the final total. The existing USD 400
accrued/non-compute reserve and the USD 800 ceiling remain in effect.

All eight previous trial VMs were deallocated. The unused Flexible Server source
was unexpectedly Ready after its earlier stopped state; recent activity only
showed resource-health changes, not a user start action. Its cause is not
asserted. It was stopped again and Stopped was confirmed. The other five
Flexible Servers remained Stopped. Old failed runners/targets were not started.

Only the retained PostgreSQL source VM was started for r3. Its initial fresh
health check showed disk 9%, zero swap, zero boot OOM events and valid certificate
chain / more than 96 hours remaining. Its intentionally restart-disabled
PostgreSQL container was stopped; an explicit start of that same container was
submitted without changing data, credentials or certificate validation.

## Readiness passed; private credential entry pending

The new VM deployment completed at `2026-09-14T04:59:04.242274Z` and the GUI
reconciled it without replay. Guest readiness operation
`091c9cda-423e-4306-ba8c-f0b7fdd9012a` was submitted at `05:00:23.758Z` and
completed successfully. Version, commit and archive SHA match the pinned build;
`postgresql-native-floats-v1` is advertised. Boot ID is
`842f6400-ed79-40c0-a7f8-c3fa38e67bcc`; idle=true, disk 3.48%, swap=0, OOM=0.

The source container start also succeeded. Current certificate validation
accepts both its DNS name and private IP, with more than 96 hours remaining;
source disk remains 9%, swap=0. No data or credential changes occurred.

The installed GUI re-reviewed the source, accepted the already-covered complete
inventory read approval and opened **Read-only source password**. The operator
must enter the existing read-only credential in that private input and press
Enter. No inventory intent exists yet; its operation is created only after
credential entry and fresh health admission. No password is stored in this
report, the source form or a generated LoadJob. The source and new runner remain
running within the renewed trial window; old failed resources remain stopped.

## New inventory started

After operator input, the GUI refreshed idle readiness at
`2026-09-14T05:04:42.136Z` and submitted new inventory
`fe5812fd-f7c1-4035-9aeb-a4868a1980cd` at `05:04:52.004Z`. The guest accepted
and started it on the same boot with configuration SHA-256
`0ea9b7674c9e81f1c237a0856675113437be06970cea5ceebf992e45c01cb43e`.
The GUI has reconciled `running`, not terminal success. No old job was replayed.
Current running VMs are only the r3 runner and retained PostgreSQL source;
no recent Policy-category events were returned by the resumption query.
Cost Management was throttled on refresh (429); the prior lagged total and
USD 400 reserve remain, with the unchanged USD 800 / September 16 deadline.

## Inventory complete and imported

The full source inventory completed at `2026-09-14T05:07:22.012078968Z`.
All 18 mappings reached EOF in one repeatable-read snapshot: 1,600,000
vertices and 4,000,000 edges. Outcome and both checks pass, with empty errors
and incomplete checks. Estimated target storage range is 3,484,790,000 through
9,900,772,000 bytes, before the plan's additional headroom.

The GUI create-only export and subsequent import verified 2,947 bytes against
SHA-256 `4ef1cbc69c6a5777fb3fb8de5503db66361bd4a347ffdbae7e02f7f55909a60a`.
The installed UI displays **Hash-verified source report**. This is inventory
success, not a migration pass. Initial target review correctly refused stale
guest health without submitting a deployment; readiness is being refreshed.

## Fresh private target submitted

Fresh GUI readiness passed at `2026-09-14T05:14:40.132Z`. Target review and
Azure preflight then passed. The GUI saved the secret-reference-only LoadJob
and plan in the existing private trial output folder and submitted deployment
`afpg-c275d043de934b0ab2b0` once. Azure reports Running at
`2026-09-14T05:16:35.795248Z`; no deployment was replayed.

- New server: `afpg-c275d043de934b0ab2b0`, PostgreSQL 18 / AGE.
- Japan East, zone 1; Standard_D4ds_v5, 128 GiB; trial HA disabled.
- New delegated subnet `10.246.11.0/24` in the existing trial VNet, private DNS,
  public access disabled, no peering or source firewall changes.
- Same runner will resize separately to Standard_D4s_v5; loader RSS bound 4 GiB.
- Plan hash: `f0bdbedef7a152b630073cf9c62489d7f3e99b1e903294b2a803375e050ba18d`.
- Combined target/runner compute quote USD 0.736/hour, USD 400 reserve,
  unchanged USD 800 ceiling and September 16 deadline. Other retained/source
  resource charges remain covered by the aggregate trial budget, not this quote.

AGE preload readiness and same-VM resize are still required after provisioning.
No migration job has been created. Previous failed graphs remain unchanged.

## Target provisioned and same VM resized

Azure deployment succeeded at `2026-09-14T05:23:41.224719Z`; the installed GUI
reconciled `provisioned`. It then submitted the dedicated target's AGE preload
restart at `05:24:57.134Z` and reconciled `finished` with Ready state and no
pending preload restart. Public access remains Disabled.

GUI readiness refreshed at `05:25:15.184Z` before resize. Same-VM resize
started at `05:26:52.649Z`: deallocate, reconcile, change SKU, reconcile,
start, reconcile. The GUI reports `finished` at Standard_D4s_v5. Preserved
disk/NIC/identity/placement SHA-256 is
`7062f043e5ce063f72d44eaa65d2bc6fc0f5c013c04f743854c5ff4407ea27b2`.
Source VM and failed targets were not resized or modified. Recent governance
events include target deployIfNotExists evaluation; no override was applied.
Post-boot guest readiness is the next admission gate, before a new migration.

Post-boot GUI readiness passed at `2026-09-14T05:31:00.055Z`, on new boot
`0e859515-6c61-4620-b069-198082fd3daf`. Pinned commit/archive and native-float
capability match; idle=true, disk 3.48%, swap=0, OOM=0. The new migration
preflight passed and the already-covered native start approval was accepted.
The remaining private input is the read-only PostgreSQL source password for
this migration, not a replay of the inventory. No old target or job is reused;
the new durable job will be created only after credential entry and admission.

## Admission timing failure retained before job creation

After private credential input, idle readiness refreshed at
`2026-09-14T05:38:21.500Z` on the same boot, disk 3.48%, swap/OOM zero.
The second migration preflight refused the sized VM's ARM readiness. No
migration intent/job was created and no target writes were submitted. Subsequent
read-only VM inspection confirms matching workflow, Standard_D4s_v5, Running
and Succeeded. The readiness Run Command completed in Azure at 05:38:54.875Z.
An ARM state-settling race is suspected; the original rejected VM response was
not retained, so its exact transient provisioning state is not asserted.

The Extension now polls GET-only for at most 30 seconds when the same running,
correctly sized/owned VM specifically reports Updating. Other provisioning or
identity changes fail immediately. The final Succeeded and fresh health/budget
checks remain mandatory; no migration or readiness write is replayed. Added
regressions cover successful settling, bounded timeout, changed identity/size/
power, terminal failure, and health/budget expiry. Typecheck, build and all
183 unit tests pass. Updated VSIX SHA-256:
`675e03c6d24c5a7582900c061f19081407a060cde901700ca9ee83fd280f120c`.
Installed and reloaded in the real VS Code; Linux artifact remains unchanged.

The saved r3 workflow was reconnected through the GUI without replay. New
readiness at `05:44:21.931Z` confirms the same boot, idle=true, disk 3.48%,
no swap/OOM. The updated migration preflight passed and the native start
confirmation was accepted. The actual UI is again at **Read-only PostgreSQL
source password**; previous credentials were discarded, not retrieved or saved.
No migration intent exists yet. Only the source and r3 runner VMs are Running;
the r3 target is Ready and all six previous Flexible Servers remain Stopped.

## Fresh migration running

After the next private input, the GUI submitted fresh durable job/operation
`b4e66d41-cfdc-4bf3-bc84-2181a7ff5a37` at `2026-09-14T05:46:08.973Z`.
The installed GUI reconciled accepted, then running without replay. This job
uses the same approved r3 target, fixed native-float Linux archive and boot
`0e859515-6c61-4620-b069-198082fd3daf`, with the complete 18-label inventory.
The PostgreSQL source and target use verified TLS. Credentials are absent from
the report and retained workflow metadata. Full qualification is still pending.
Cost Management again returned 429; prior lagged costs and the conservative
USD 400 reserve remain under the unchanged renewed USD 800 / September 16 gate.

## Complete counts PASS; full P1 verification submitted

The retained report generated at `2026-09-14T05:51:26.657388413Z` passes all
24 checks and all 18 exact label counts, with 5,600,000 records, zero rejects,
empty errors and empty incomplete checks. From GUI submission through that
report is about 5m18s, including preparation/load/count verification; this is
not isolated loader throughput. The GUI imported and verified 9,619 bytes:
SHA-256 `dc829bd1aa2e48ab7e858c37389ce02691e18107cd9754de03b0a7c6e505738b`.
Job fingerprint is `3a5aea659bf18ff0969206b56bd4c2101b65f21f4146020caad36cbc73aa94e6`.
Its visible counts result is PASS, explicitly distinct from property validation.

Post-load readiness at `05:56:21.889Z` confirmed the same boot, idle=true,
disk 3.53%, no swap/OOM. Target storage peaked around 14.70%, below 80%.
The earlier read-only diagnostic found the worker active with cgroup memory
51,675,136 bytes under its 4 GiB cap and disk 4%; its optional JSON extraction
failed because jq is absent on the VM. This diagnostic was not a job failure.

The installed GUI selected the already-reviewed frozen P1 verifier (commit
`19026db1930a7893ac4fb30f8647e1c277fe9920`, archive SHA-256
`8e9bf7ec6c37aa06b5aa49fd204663c0abd723c06eda8655631e9d2f776d2c49`).
It submitted independent verification `e79bb594-4af6-4c7a-9fca-172e166f46dc`
at `05:58:43.923Z` for this exact migration job. All 64 ranges / 5.6M records
must match; qualification remains pending. No source, graph, loader or network
changes are made by that verifier.
