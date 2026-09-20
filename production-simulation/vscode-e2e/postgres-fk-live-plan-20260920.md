# Isolated PostgreSQL FK GUI qualification proposal

Status: bounded fixture/resource scope approved by the user on 2026-09-20;
GUI setup in progress. No source writes or compute starts yet. Exact storage
access confirmation is pending; this is not a live qualification result.

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
- The storage confirmation proposes `af24bd714a70ee486582db90` with account-only
  Blob Data Contributor for the signed-in user. Creation is not yet submitted.
- Fresh cost query returned HTTP 429; the last confirmed delayed RG month-to-date
  total is USD 295.608869711763. The USD 800 ceiling remains unchanged.
- Source is stopped with public access disabled; RG locks are absent. The
  30-minute compute clock has not started. Do not use the old session's deadline
  or restart its completed workflow.
- The create-only fixture SQL is retained beside this plan. It uses one bounded
  transaction, fails if the schema already exists, creates no new role/password,
  and grants only USAGE plus SELECT on the two new tables. It has not run.
