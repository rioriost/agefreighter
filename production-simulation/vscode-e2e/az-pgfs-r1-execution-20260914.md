# AZ-PGFS r1 installed-GUI qualification

Status: source configuration saved and reviewed in the installed VS Code GUI.
The user approved dedicated storage; its deployment and artifact upload passed.
Runner placement passed after the region-name fix below; the installed GUI
submitted the private discovery VM deployment once. Guest readiness passed;
the complete inventory passed and its hash-verified report was imported through
the GUI. A fresh private target is provisioned, AGE restart and same-VM resize
are complete. Migration, complete counts verification and independent full
canonical verification passed in the installed GUI. Overall qualified route
coverage is now 6/9. The chronological sections below preserve earlier gates.

## Scope and preserved source

- Workflow: `29558917-403e-4a76-aaa0-de07122ea9c6`.
- Source selection: PostgreSQL / Azure, then the approved subscription,
  existing trial resource group and ARM-discovered Flexible Server candidate.
- Retained PostgreSQL 18 source: `afpg-p1-source-20260907`, database `p1source`,
  read-only user `agefreighter_reader`; no credential is included here.
- Private networking and TLS certificate validation remain enabled. No source
  public endpoint or firewall change was made; the Linux runner uses the
  existing trial VNet's runner subnet.
- Source data was not recreated. The frozen P1 fixture has 1,600,000 vertices
  and 4,000,000 edges; fresh GUI inventory and final verification remain required.
- All 18 saved GUI mappings exactly match `fixtures/postgresql-p1-mappings.json`.
- SHA-256 of the JSON-serialized reviewed source configuration:
  `b1d422780ada2953cfd6b252331a86c97c94138df6111568ad247ad3bf89f8f4`.

## Gates and current handoff

At the initial preflight all ten trial VMs were deallocated, all eight Flexible
Servers were stopped, no resource-group locks were present, and the recent
activity query returned no policy actions. The renewed ceiling remains USD 800
and the deadline remains `2026-09-16T07:14:35.311Z`; neither is reset by this route.
The last returned, delayed cost was USD 36.7436452084183, not final billing.

Only the retained AZ-PGFS source was started and subsequently observed Ready.
A concurrent expiry-tag update returned `ServerIsBusy`; a later separate tag
update succeeded, and a fresh read confirmed the renewed September 16 deadline.
The start was not replayed. All other sources, targets and runners remain
preserved. Source compute and retained storage charges continue while waiting.

The user approved creation of storage account `af29558917403e4a76aaa0de`
in the existing trial group. Its explicit confirmation describes a grant of
Storage Blob Data Contributor to the signed-in user on this new account only,
an HTTPS network-public endpoint, disabled anonymous access and disabled shared
keys. Deployment succeeded and the GUI reconciled it. Its initially disabled
public network was changed only on this trial storage using the previously
authorized `SecurityControl=Ignore` exception; HTTPS/TLS 1.2, disabled anonymous
access and disabled shared keys were independently confirmed. No source server
is exposed. GUI upload verified the unchanged Linux archive and marked it ready.

## Live preflight defect and regression fix

The first GUI runner preview correctly made no deployment but incorrectly
rejected matching source/runner regions. A GET of the source using API
`2024-08-01` returned `location: "Japan East"`, while the placement catalog
uses `japaneast`. Commit `dd6b401` normalizes the source region's whitespace
and case before comparison. Wrong/missing regions, zone mismatches and
cross-subscription placement remain blocked.

All 189 unit tests, typecheck, build and package passed. The updated 2.4.0
extension was installed on this Mac and the window reloaded without changing
the saved draft or Linux artifact. VSIX SHA-256:
`51677a1732f2cb5fb4984b654340d3175a2d96f8b9c3d2823a35d5511faa953b`.

## Remaining sequence

1. Scoped storage and the reviewed Linux artifact upload are complete.
2. Reconcile the private B2s_v2 discovery runner; verify guest
   readiness, artifact, storage, idle state and swap/OOM.
3. Obtain the source password privately through VS Code; run complete inventory
   and import the checksummed result through the GUI.
4. Review sizing, save the LoadJob and provision a fresh private AGE target;
   resize the same runner and refresh readiness.
5. Run migration and strict complete verification, then compare all 64 P1
   digest ranges and the canonical root through the GUI and independently.
6. Only on full qualification, update route coverage; preserve evidence and
   stop/deallocate this route's compute resources.

The intended Linux artifact remains `2.4.0-dev.9ef16968363b`, archive SHA-256
`10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
The expected P1 canonical root remains
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.

## Runner deployment

The repaired GUI preflight/what-if passed and reviewed only new workflow-owned
resources. Runner `af-29558917403e4a76aaa0` was submitted once, with deployment
observed Running at `2026-09-14T09:20:04.677013Z`. Japan East / zone 1,
B2s_v2, compute USD 0.109/hour plus storage/network. Preview hash:
`ece8a9e57f4597edd09f04823bc3f97c70c5acdc126e1138675510b83efd99a5`.
No public IP, SSH ingress, peering or source firewall change was requested.
The VM identity receives Blob Reader only on the synthetic-test artifact
container. At this stage the agent was still initializing; ARM VM creation
is not evidence of guest readiness or successful assessment.

Cloud-init subsequently reported done with the runner tools present. The GUI
readiness result at `2026-09-14T09:22:35.060Z` confirms the pinned artifact,
the native PostgreSQL floating-point capability, an idle runner, 3.5079% disk
usage, zero swap and zero OOM events. The saved 18 mappings were reviewed again.
The approved complete-inventory action is paused at **Read-only source password**
for the retained Flexible Server's `agefreighter_reader` account. No password
was read into this conversation and no source assessment has yet been dispatched.
Only this new runner and the AZ-PGFS source are running; old route resources
remain preserved and stopped/deallocated. The budget and deadline are unchanged.

## Complete inventory and target plan

The user entered the read-only password privately. Operation
`e0a624da-95ae-416a-a635-e81898e307d1` was submitted once at
`2026-09-14T09:25:35.439Z`. Its report generated at
`2026-09-14T09:27:50.558680111Z` passes both checks, with no errors or incomplete
checks. All 18 mappings reached EOF in one repeatable-read snapshot: 1,600,000
vertices and 4,000,000 edges. Counts are not full migration verification.

The GUI exported and imported the exact 2,947 bytes; independent local hashing
agrees with SHA-256
`ea1bc40b73047cc683ea6edc385380c499a5de4e76abb8d5e5b1430873afa4fe`.
Mapped bytes: 553,598,000; estimated target storage range:
3,484,790,000–9,900,772,000 bytes. Post-inventory GUI health at
`09:30:53.120Z`: idle, same boot/artifact, 3.5093% disk, swap/OOM zero.

The saved plan proposes `afpg-29558917403e4a76aaa0`, PostgreSQL 18 / AGE,
GP D4ds_v5, 128 GiB, HA off, Japan East zone 1. A fresh delegated subnet
`10.246.13.0/24` does not overlap any existing trial subnet; private DNS remains
inside the runner VNet. No public database access or peering is requested.
The same runner's later resize is D4s_v5, with a 4 GiB application bound.
Plan SHA-256: `c214d8eb139f2071b1a9a16c85aa9529113d99763b326deb527e206f0207184b`.
The GUI saved a secret-reference-only LoadJob and target plan in the private
AZ-PGFS staging folder. Target credentials remain in VS Code SecretStorage.

Reviewed compute is USD 0.736/hour plus USD 400 accrued/non-compute reserve;
the USD 800 ceiling and September 16 deadline are unchanged. A fresh cost
query returned HTTP 429 and was not retried; the earlier USD 36.7436452084183
observation remains delayed rather than a current bill. The recent policy
activity query returned no actions and the trial group had no locks.

## Target and same-VM resize completed

Target deployment was observed Running at `2026-09-14T09:33:33.080213Z` and
all children subsequently succeeded, including database, AGE allowlist and
preload settings. The installed GUI reconciled the target to provisioned.
The serialized child-write fix avoided the previous route's `ServerIsBusy`:
no repair, target recreation or deployment replay was needed.

The separate AGE restart was submitted at `09:41:29.703Z` and reconciled to
finished. Live configuration is `pg_stat_statements,age`, with restart pending
false. Same-VM resize began at `09:42:58.203Z`; the GUI separately controlled
deallocate, resize and start, then reconciled finished. The NIC, persistent
disk and identity preservation hash is
`c55f7f9790ffdca20b456dc1fcb22e68ab66bdc58ece9745fc740676f973849a`.

Post-resize GUI readiness at `2026-09-14T09:46:39.122Z` confirms the new boot,
unchanged Linux version/archive/capabilities, idle state, 3.5103% disk usage,
zero swap and zero OOM events. The source VM was not resized (the source is
Flexible Server). The trial group still has no locks and the recent policy
activity query returned no actions. Source password is not retained; a new
private entry is required for the separately approved migration.

## Migration and complete counts verification

After private source-password entry, the GUI submitted durable job
`958d33c4-b7a9-449e-9017-04f7081a23a9` once at
`2026-09-14T09:49:32.699Z`. The verification report generated at
`09:54:50.418587147Z` passes all 24 checks and all 18 exact label counts:
1,600,000 vertices and 4,000,000 edges, zero rejects, no errors or incomplete
checks. Elapsed submission-to-report time is approximately 5 minutes 18 seconds.
Configuration fingerprint:
`0705a346c5894657904e73fc57d753b48262e9f06f23af249ba8a9de89e38f52`.

The installed GUI exported and imported the report and displays **Exact source
and target counts agree with no rejects**. Independent local hashing confirms
9,619 bytes and SHA-256
`88fe5d0fbabd944e4852c75f9d5023d2158d16aac54530fb477dd7ba9272a87d`.
No migration replay, graph repair or source modification was performed.

Post-load readiness at `09:57:02.197Z`: idle, same boot and loader artifact,
3.5127% disk, zero swap and zero OOM events. There are no group locks; fresh
policy events are audit/auditIfNotExists observations, not enforcement changes.
The unchanged USD 800 / September 16 authorization and conservative reserve
remain in force.

The GUI submitted independent read-only P1 qualification
`7f5badd9-35e3-49b5-bb44-11f7701cd911` at `09:58:36.376Z` using the same
pinned verifier as the previously qualified routes. This compares every typed
property, identity and endpoint across all 64 canonical ranges; counts alone
do not qualify this route.

## Full canonical qualification PASS

The report generated at `2026-09-14T10:00:50.732774263Z` was exported,
hash-verified and imported through the installed GUI, which displays
**P1 full canonical digest: PASS**. All 1,600,000 vertices and 4,000,000 edges,
64 ranges, typed properties, identities and endpoints agree with the frozen
fixture. Independent local comparison checked every leaf and recomputed both
canonical roots; each equals
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.

Result: 23,218 bytes, SHA-256
`39f4898473a7c639d322ec0ea129556b352012c2e365a15600de06b0968bdf82`.
Verifier commit `19026db1930a7893ac4fb30f8647e1c277fe9920`; archive SHA-256
`8e9bf7ec6c37aa06b5aa49fd204663c0abd723c06eda8655631e9d2f776d2c49`.
No loader replacement, target repair, migration replay or source data change
was needed. This is the P1 guided Azure Flexible Server source path, not a
new production-scale qualification or a guarantee for arbitrary source schemas.

Extension typecheck and all 189 unit tests pass. Observed target storage maximum
was 14.7127%, below the 80% limit. Budget and deadline are unchanged.
[Redacted qualification evidence](evidence/az-pgfs-r1-p1-pass-20260914.json).

Final health at `10:02:09.739Z` confirms idle state, unchanged loader/boot,
5.0942% disk usage, zero swap and zero OOM events. Only this route's runner
was deallocated and its source and target Flexible Servers stopped. At
`10:05:39Z`, all eleven trial VMs were deallocated and all nine Flexible
Servers were Stopped. Nothing was deleted; all jobs, databases, reports and
failure evidence are retained. Storage charges continue, and Flexible Servers
automatically restart after seven days unless managed before then.
