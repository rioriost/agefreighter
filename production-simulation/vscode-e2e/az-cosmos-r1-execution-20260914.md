# AZ-COSMOS r1 installed-GUI qualification

Status: **migration and complete counts PASS; independent P1 verifier failed**.
AZ-COSMOS is not qualified. The failure is not yet proven to be a data mismatch;
no final comparison report was produced. All job and guest evidence is retained.
Earlier handoffs below are retained as history.
Overall qualification remains 6/9; the earlier headless Cosmos inventory is
not a guided migration pass.

## Scope and live preflight

- Workflow: `7b79f05d-1dc1-40a6-b3dc-6c8129d4e0c1`.
- Installed VS Code GUI selection: Azure Cosmos DB for NoSQL / Azure,
  approved subscription / `rg-af-vscode-p1-20260905-a`, then Discover and
  account `afcosmosp120260907`.
- Runner placement: Japan East / zone 1, initial B2s_v2, retained private
  trial VNet runner subnet. The Cosmos data region is checked separately from
  its account metadata location; no source logical-zone equivalence is claimed.
- Source account: provisioning Succeeded, public network access Disabled,
  local/key authentication disabled. Container `p1/graph` remains at autoscale
  maximum 4,000 RU/s. The retained fixture has 1.6M vertices / 4M edges; fresh
  GUI inventory and full target canonical verification are still required.
- No group locks were returned. Budget remains USD 800 and the renewed
  deadline remains `2026-09-16T07:14:35.311Z`; neither is reset for this route.
  A fresh Cost Management query returned HTTP 429 and was not retried.
  The last returned USD 36.7436452084183 is delayed billing, not final spend;
  retain the conservative USD 400 accrued/non-compute reserve.

The matching published 2.4.0 Linux release is unavailable. The GUI correctly
created only a local draft and submitted no VM deployment. The same reviewed
development loader used by AZ-PGFS will be selected after transfer storage is
prepared; do not substitute an unreviewed build.

## Current GUI handoff

Current action: preserve the failed verifier and stop route compute while awaiting
approval to run an instrumented read-only verifier. Do not replay this operation,
patch the committed graph, or classify counts-only success as qualification.
The following paragraphs describe the previous storage-approval handoff.

The source form has unsaved basic entries for `az-cosmos-p1-r1`, namespace
`p1`, host `afcosmosp120260907.documents.azure.com`, database `p1`.
All explicit P1 mappings still need to be entered and reviewed. No password or
account key is required; a later, separately reviewed assignment will grant
only Cosmos Built-in Data Reader to the new owned runner identity.

The native confirmation proposes storage account `af7b79f05d1dc140a6b3dc6c`
in the existing trial group, Japan East, Standard LRS. It grants the signed-in
user Storage Blob Data Contributor on **this new account only**. Its HTTPS
endpoint is network-public; anonymous access and shared keys remain disabled.
It does not expose the Cosmos source. The agent paused without selecting
**Create storage and scoped role**, because this adds access on a new scope.

## Remaining sequence

1. Approve and reconcile dedicated storage; preserve secure defaults and use
   only the already authorized trial-storage exception if policy requires it.
2. Upload the unchanged, hash-pinned Linux artifact; approve and deploy the
   private discovery runner; verify guest readiness and budget/health gates.
3. Review all 18 explicit Cosmos P1 mappings and the read-only identity grant.
   Preserve the source-immutability window: Cosmos does not provide one
   transactional snapshot across the complete mapped source.
4. Run complete inventory, import its hash-verified report, review sizing and
   provision a new private AGE target; resize the same runner.
5. Migrate once, require complete counts verification, then compare all 64
   P1 ranges / typed properties / identities / endpoints and canonical root.
6. Only on full GUI qualification, update coverage and stop route compute;
   preserve all data and evidence. No P4, release or unrelated deletion.

## Storage, pinned runner and reviewed mappings

The user completed the storage approval in VS Code. Its deployment succeeded
and the GUI reconciled it to ready. Policy initially disabled public network
access; the previously authorized trial-storage-only `SecurityControl=Ignore`
exception and the unchanged September 16 expiry tag were applied to this
account, then authenticated HTTPS connectivity was enabled. TLS 1.2, disabled
anonymous access and disabled shared keys were independently confirmed.
No source firewall or Cosmos authentication setting changed.

The GUI uploaded and verified the same 37,056,164-byte Linux archive used by
the previous qualified route: version `2.4.0-dev.9ef16968363b`, commit
`9ef16968363b31214324f392553f7c8e88150272`, SHA-256
`10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
No binary was rebuilt or substituted.

The reviewed VM preview passed; hash
`4ed3666c5240cffa4436ea437d06a65037108e8473062ef576f4bbb45fdc70b0`.
The GUI submitted `af-7b79f05d1dc140a6b3dc` once: Japan East / zone 1,
Standard_B2s_v2, compute USD 0.109/hour plus retained storage/network charges.
It uses the existing private runner subnet, no public IP or SSH ingress, and
Blob Reader on this workflow's synthetic-test artifact container only.
The GUI reconciled ARM provisioning and then verified Linux readiness at
`2026-09-14T12:00:25.914Z`: correct version, hash and Cosmos capabilities,
idle, 3.4785% disk, zero swap and zero OOM events. The new VM is running;
the earlier eleven trial VMs and nine Flexible Servers were not started.

All nine vertex and nine edge mappings were entered through visible GUI fields
and reviewed. An independent comparison confirms exact equality with
[the Cosmos P1 mapping fixture](fixtures/cosmos-p1-mappings.json), derived from
the frozen PostgreSQL P1 properties/endpoints with collection `graph` and no
SQL schema. Database is `p1`; identities and endpoint fields are also explicitly
preserved as graph properties where required. The saved configuration SHA-256
is `f05eb6d79526a1d8284d7050579d3a173660666b8e984826f3ac0dba3e52007a`.

Fresh source checks confirm actual data region Japan East, Private Endpoint
Approved, public network Disabled and local/key authentication disabled. No
group locks or recent policy activity were returned. No source data was changed.

The next native GUI confirmation grants **Cosmos DB Built-in Data Reader**
to runner identity `e654ffc2-5287-46d8-91a9-758c654002dc`, scoped only to
`afcosmosp120260907`. Assignment intent
`d203e078-1b89-4aa7-8654-1de59e8ac86c` is previewed, not submitted. It grants
no writes, uses no account key and does not expose the source. The agent
paused at this new access-grant type; source inventory has not started.

## Data Reader approved; complete inventory started

The user approved the scoped Cosmos Data Reader assignment at
`2026-09-14T12:05:17.371Z`. The installed GUI reconciled the exact assignment
to ready; no Contributor role, account key or public access was introduced.
The same 18 mappings and source-immutability condition were reviewed again.
Fresh automatic readiness at `12:06:16.851Z` confirms the same boot/artifact,
idle state, 3.4787% disk, zero swap and zero OOM events.

The installed GUI submitted complete inventory operation
`faa15c58-8f07-4f45-b006-98468ca1b1d4`, bound to configuration SHA-256
`f05eb6d79526a1d8284d7050579d3a173660666b8e984826f3ac0dba3e52007a`.
It was subsequently observed running. Limits remain 30 minutes / 4 GiB /
no swap; Cosmos source data remains unchanged. No target writes have begun.
The source still uses the retained 4,000 RU/s autoscale maximum. Source count,
capacity acceptance, migration and independent canonical verification remain
separate gates; this running inventory is not a qualification pass.

The recent policy modify event was the transfer storage at `11:52:38Z`, before
the already recorded scoped exception; there were no group locks. This is not
evidence of a Cosmos source configuration change. Budget/deadline are unchanged.

At approximately four minutes of guest runtime, a read-only diagnostic confirmed
the inventory still running, RSS 27,800 KiB, zero swap and 4% root-disk usage.
Azure request metrics also show continuing reads. The first diagnostic used
unavailable `jq`; it was rerun with standard read-only tools and did not change
the running inventory. No configuration or throughput tuning was applied.

The nine GUI field-to-CLI configuration contract tests passed with a freshly
built isolated local test CLI. The first invocation lacked the required test
binary environment variable; supplying the test harness prerequisite resolved
that setup failure. This local validator test did not replace the Linux source
inventory or execute a local migration.

## Inventory accepted; private target submitted

The inventory completed at `2026-09-14T12:15:22.217304956Z`, approximately
8 minutes 49 seconds after guest start. All 18 labels reached EOF: 1,600,000
vertices and 4,000,000 edges. Read-only and source-counts checks pass, with
no errors or incomplete checks. The installed GUI exported and imported the
2,940-byte report; independently verified SHA-256:
`f6056bc71f83c1ba75510b3fb28b3d662550d23e83df6c27f431297f155686fe`.
Mapped records total 458,398,000 bytes; the reported storage estimate is
3,008,790,000–8,567,972,000 bytes. No source throughput tuning was applied.

Readiness at `12:19:04.406Z` passes: idle, 3.5104% disk, zero swap/OOM.
The GUI saved the secret-reference-only LoadJob and target plan in the local
trial staging folder and submitted target deployment
`afpg-7b79f05d1dc140a6b3dc` once. Preview SHA-256:
`bc20f6499fd2492e9de03d019851d3abd7196f593bc48ddbd7e3505bd63bcb3d`.
It provisions PostgreSQL 18 / AGE, D4ds_v5, 128 GiB, Japan East / zone 1,
with public access disabled and dedicated subnet `10.246.14.0/24` in the
existing VNet. The same runner's reviewed migration size is D4s_v5.
Combined target/runner compute estimate is USD 0.736/hour; USD 400 additional
reserve, USD 800 ceiling and September 16 deadline are unchanged.
The GUI reports target `submitted`; ARM completion, AGE readiness, migration
and full canonical verification are separate remaining gates.

## Target ready, same-VM resize complete; migration started

The GUI reconciled the single target deployment to provisioned and separately
completed the AGE preload restart (submitted `12:29:17.392Z`). The first resize
attempt was correctly blocked by stale guest health, before any resize request.
After refreshing readiness, the GUI explicitly deallocated, resized and started
the same runner as Standard_D4s_v5, preserving its disk, NIC and system identity.
Resize preservation SHA-256:
`18f89135c466e5ce92b391b46d20a203310e6748446c12e408cad1623f6da04d`.
Post-boot readiness at `2026-09-14T12:36:23.496Z` passed with the same pinned
artifact, idle, 3.5115% disk, zero swap/OOM. New boot ID:
`fd1127f2-d3f0-4183-a5cb-d577d20e3c4c`.
The target is Ready with public access Disabled. No group locks were found;
target policy deployIfNotExists activity was observed and retained as governance
context, not treated as a user-requested policy change.

The GUI submitted one Cosmos migration plus complete counts verification at
`2026-09-14T12:37:48.928Z`: durable job/operation
`7fa558e4-8027-4335-9a2b-564f70b3df02`. It is bound to the accepted inventory,
same mappings and pinned Linux loader. Cosmos authentication uses the retained
read-only managed identity; no source keys or passwords were requested. Target
credentials remain in SecretStorage/protected transport. No migration replay,
source modification or new authorization window was introduced.

## Migration and counts PASS; full verifier failure retained

The Linux migration sequence ran from `2026-09-14T12:38:04.480829277Z` to
`12:48:17.305124833Z` (about 10m13s). The installed GUI imported the 9,619-byte
counts report generated at `12:48:16.350782052Z`; independent byte/hash validation
passed. SHA-256:
`740452a3a419ad15bfe7b8f72cdb73cb28d4f79a34d9141d81498c305e657d64`.
All 24 checks pass with zero errors, incomplete checks and rejects; all 18 labels
agree. Job fingerprint is
`ecf0f4e6fca7a2963d07b42338a7023198a57d31148907697a952548934171c3`.

The GUI submitted full P1 operation `cb0c7805-5d5e-4ccd-bff8-1d39b6015b0f`
at `2026-09-14T12:54:26.352Z` with the unchanged verifier archive
`8e9bf7ec6c37aa06b5aa49fd204663c0abd723c06eda8655631e9d2f776d2c49`.
Preflight was idle, 3.5298% disk, zero swap/OOM. The GUI later reconciled
**failed**, never PASS. The guest retains its executable, archive, generated
fixture manifest, empty stdout and 72-byte generic stderr; no `result.json`
exists. This proves verifier failure, not its precise cause or a canonical
root mismatch. No failed operation was replayed.

Code inspection identifies an ordering assumption worth testing: target digest
reads by allocated graph ID but demands increasing fixture source keys; the
generated Cosmos SELECT has no ordering clause. This is a hypothesis, not a
live-confirmed cause. A local diagnostic-only change now records fixed,
secret-free failure stage/code identifiers in a create-only private
`failure.json`, including a distinguishable source-key-order error. Existing
root, row, property and endpoint acceptance checks remain unchanged. The
verifier and rangedigest unit suites pass. This new diagnostic code has not
been deployed or run against Azure; the original verifier/evidence is unchanged.

At `13:01:25Z`, a read-only guest check found no loader/verifier processes,
zero swap and 6% disk use. The observed target storage maximum was 15.2576%,
below the 80% gate. Cosmos still has public access Disabled, key auth disabled,
Japan East data placement and maximum 4,000 RU/s. No source data was rewritten.
After confirming resource ownership and no locks, stop requests were issued
for only this route's runner and Flexible Server; no resources or evidence
were deleted. Final stopped-state confirmation is recorded below when available.

At `2026-09-14T13:04:29Z`, all twelve trial VMs are deallocated; nine Flexible
Servers are Stopped and this route's server is Stopping. Cosmos provisioned
throughput and retained storage still incur charges. Flexible Server automatic
restart after seven days remains relevant. The USD 800 ceiling and September 16
deadline are unchanged. `go test ./production-simulation/...` passes; no revised
verifier binary has been installed or used for qualification.

Next: after approval, retain the failed operation and its active-marker evidence,
run a separately identified diagnostic without replaying migration, determine
the exact cause, review any correction, and requalify the unchanged job or
explicitly review a fresh migration if a loader defect requires it. Never
relax the canonical checks to turn this failure into a pass.
