# Isolated PostgreSQL FK GUI qualification proposal

Status: bounded fixture/resource scope approved by the user on 2026-09-20;
Dedicated storage, pinned runner readiness and isolated fixture setup succeeded.
The installed GUI reconciled the successful catalog read after private password
entry. Exact sealed report transfer/import approval is pending; no candidate or
mapped inventory is accepted yet. This is not a live
qualification result; stop both compute resources by 18:42 JST at the latest.

## Why a separate fixture is needed

The accepted P1 source has no foreign keys. Adding constraints to its tables
would change the qualified source and is not necessary. Local PostgreSQL catalog
tests already cover FK and conservative rejection rules; actual installed-GUI
discovery, imported proposals, explicit adoption and mapped inventory still need
live evidence with real constraints.

## Proposed bounded scope

1. After explicit approval and fresh cost/governance checks, start the existing
   source and provision one new private B2s_v2 discovery VM with workflow-owned
   transfer storage in the existing test RG/network, for at most 30 minutes from
   the first compute start request. Preserve the USD 800 ceiling. Obtain the
   exact GUI preview/resource identities and artifact approval before deployment.
   Do not create a target or run a migration.
2. In `p1source`, create a new dedicated schema `af_fk_qualification_20260920`
   only if absent. Stop on an existing schema rather than reusing/overwriting it.
   Two tiny synthetic tables: two suppliers and three products, bigint primary
   keys; products have one validated, non-null supplier FK and a second nullable
   supplier FK. No P1 tables, rows, constraints or accepted graphs are changed.
3. Grant the existing reader USAGE on this new schema and SELECT on its two
   tables only. This is an explicit new privilege scope requiring approval;
   credentials, role attributes and existing grants stay unchanged.
4. Use a fresh isolated workflow/operation under the product's normal reviewed
   runner provisioning rules. Do not reset, edit or repurpose completed operator
   records. Review confirmed that `runnerNames` derives resource names from the
   workflow ID, preflight requires proposed resources to be absent, and
   `startCatalog` rejects a retained catalog/assessment/target. Reusing a completed
   workflow or rebinding an existing VM is not an available qualified route.
   No cloned records, reuse implementation or hand-written guest shortcuts.
5. Through installed GUI, approve read-only catalog discovery of this schema
   only; import the sealed report. Expect two vertex candidates, one safe FK
   edge candidate and manual-review warning for the nullable FK. Verify no
   automatic selection or adoption and no P1/out-of-schema proposals.
6. Explicitly select safe candidates with both endpoints. Review source/target
   direction, identity-only default properties and then map intended properties.
   Reload/reconnect, verify persistence, run bounded mapped inventory and import
   the sealed result: five vertices and three safe-FK edges. Counts are not a
   migration or canonical-digest qualification.
7. Stop/deallocate the exact compute immediately after terminal outcome or
   the time bound. Preserve schema, records, logs and seals; no automatic cleanup
   or broader grants. Record actual results without closing unrelated branches.

## Review / approval gate

Local review resolved the runner choice: one new workflow-owned VM/storage is
needed with the existing implementation. A new runner must not be inferred
authorized from a source-fixture approval. The user must approve the resource,
privilege and data-change scope plus bounded runtime; native action-time gates
then bind exact generated identities and pinned artifact before mutation.
This is a tiny supplemental schema-behavior trial, not another P1 graph migration.
No source write, new resource or Azure restart occurred during this design review.

## Execution preparation — 2026-09-20

- Fresh installed-GUI workflow: `24bd714a-70ee-4865-82db-90d6f4760650`,
  confirmed against the retained record filename before deployment.
- The GUI selected the existing approved subscription/test RG, PostgreSQL source,
  Japan East/zone 1, B2s_v2 and existing nondelegated runner subnet.
- Schema input is restricted to `af_fk_qualification_20260920`, database
  `p1source`, existing `agefreighter_reader`. No password has been requested.
- The user approved the storage confirmation for `af24bd714a70ee486582db90`.
  At 09:05–09:10 UTC, ARM and installed-GUI reconciliation confirmed successful
  creation and account-only Blob Data Contributor for the signed-in user.
  Anonymous access and shared keys are disabled. Public network access remains
  Disabled, so provisioning success does not establish transfer readiness.
- Fresh cost query returned HTTP 429; the last confirmed delayed RG month-to-date
  total is USD 295.608869711763. The USD 800 ceiling remains unchanged.
- Source is stopped with public access disabled; RG locks are absent. The
  30-minute compute clock has not started. Do not use the old session's deadline
  or restart its completed workflow.
- The create-only fixture SQL is retained beside this plan. It uses one bounded
  transaction, fails if the schema already exists, creates no new role/password,
  and grants only USAGE plus SELECT on the two new tables. It has not run.
- Requested approval for the established `SecurityControl=Ignore` exception and
  authenticated public HTTPS on this exact new account, plus upload of the pinned
  `d40d6ccc9a4d` Linux archive (37,197,546 bytes, SHA-256
  `2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`).
  The installed GUI displays the exact workflow-container/archive destination;
  upload confirmation is open but not submitted. No tag/network change yet.
- Storage reconciliation restored database/user defaults in the unreviewed source
  form, as in the earlier catalog trial. Re-enter and verify `p1source` /
  `agefreighter_reader` immediately before catalog review; never submit defaults.
  No credentials were entered and no source operation was submitted.

## Approved launch — 09:12–09:14 UTC

- User separately approved the exact account's tag/network exception and archive
  upload. Merged only `SecurityControl=Ignore`, preserving ownership tags; ARM
  confirmed HTTPS-only/TLS1.2, public network Enabled, anonymous/shared keys false.
  Installed-GUI upload is ready; independent Blob metadata matches the pinned
  SHA-256 and 37,197,546 bytes.
- User separately approved VM `af-24bd714a70ee486582db`, its pinned development
  installation and workflow-container-only Blob Reader grant. GUI submitted
  deployment at `2026-09-20T09:12:40.285Z` and reconciled provisioned at
  `2026-09-20T09:13:50.615Z`. B2s_v2, Japan East/zone1, USD0.109/hour compute;
  no public IP or SSH ingress. Source start was requested for the same session.
- Hard shutdown deadline **2026-09-20T09:42:00Z (18:42 JST)**, less than 30 minutes
  after the first request. Updated the safety heartbeat to this exact VM/source
  and deadline; configured the VM's Azure auto-shutdown for 09:42 UTC. Never
  extend automatically. No target creation or migration is authorized here.
- Fresh RG error activity and locks were empty. Guest readiness command
  `af-dfb8a74c-b018-40ea-b0b1-08767edfacdc` is submitted; wait/reconcile it, do not
  replay. Source still Starting at the first poll. Fixture setup helper passed
  shell syntax and Swift type checks but has not been submitted. It uses protected
  administrator parameters and the committed create-only SQL, without credential
  rotation or accepted P1 modifications.

## Fixture ready / catalog credential gate — 09:17 UTC

- Installed GUI verified pinned guest readiness at `2026-09-20T09:13:58.372Z`,
  boot `62fcab5d-c9b4-49e8-8bd6-c4779cde7770`. Guest was idle, disk 3.4811%,
  swap-used zero and OOM events zero. Source reconciled Ready/private.
- Submitted only `af-pg-fk-fixture-20260920` using protected administrator
  parameters and script SHA-256
  `d50f3af2a0675197981b0ea7cd7b0a746e54bdd9d2d97b7c8e0bbd4a4d0e688c`.
  ARM terminal execution Succeeded/exit0 with marker
  `isolated_fixture_created_scoped_grants_committed` proves the bounded SQL
  transaction and its postconditions completed. No credential rotation or P1
  mutation; the dedicated schema and its five rows are retained.
- Installed-GUI catalog confirmation displays the exact `p1source` database,
  existing reader, new runner and only `af_fk_qualification_20260920`. The user
  approved the read in VS Code; its private password box is open. Requested the
  existing PostgreSQL reader Keychain entry, never a Neo4j/admin credential.
  No catalog dispatch yet. Stop on 15-minute idle credential wait or the hard
  09:42 UTC deadline; no automatic restart or extension.
- User subsequently entered the password; installed GUI accepted catalog operation
  `511d0745-b0dd-43c7-b88f-b1144947df44`, configuration SHA-256
  `20dfdcadf0fd093633f3d959a8d6cf8193cfe082eb5d24bfde8aa1a36870a6d8`.
  Read-only guest evidence independently confirms fixture completion at
  `2026-09-20T09:16:14Z`, committed SQL log checksum
  `f89018bf71b8fa9d82803d34392fbfc6785bcf83ea31e0018550a28056192524`,
  checksum verification OK, disk4%, memory267,403,264 bytes, no swap.
- Catalog completed at `2026-09-20T09:17:57.378559415Z`, exit0, 1,966-byte
  report SHA-256 `67c10b4481d64c89ceb8b9118b7419df01740a923ceca710fdb8d974f1be8ee4`.
  Installed GUI reconciled finished and displays the sealed transfer approval.
  Requested explicit transfer/import of the synthetic two-table schema/key
  metadata to this workflow's storage and Mac; no row/password/AI transfer.
  No candidate count or safety-warning acceptance is claimed before import.
- Cost refresh succeeded: delayed RG September actual cost **USD302.628107546835**,
  under the unchanged USD800 ceiling. Hard stop remains09:42UTC.
