# Guided migration P1 qualification progress

Updated: 2026-09-13. Overall outcome: **CSV-MAC, AZ-N44 and AZ-N526 qualified; six other branches remain unqualified**.

### AZ-PGVM — storage created; transfer-network and artifact gates pending

The installed GUI selected the PostgreSQL VM through Azure discovery and saved
the TLS-verified source configuration with all 18 P1 mappings. The approved
dedicated storage and account-scoped user role are created, but Azure Policy
modified public network access to Disabled. Network-exemption and pinned test
artifact approvals remain pending. All existing VMs remain deallocated. See
the [execution sheet](az-pgvm-execution-20260913.md).
Migration and full canonical verification have not started; coverage stays 3/9.

### AZ-N526 qualification — PASS at 13:16Z

The installed VS Code 1.136.1 GUI completed the Azure Neo4j 5.26.30 branch from
source selection and exact inventory through private target deployment, AGE
preload, same-VM discovery-to-migration resize, load, counts verification and
independent full P1 verification. Job
`313f6dca-680b-4379-9eac-a7539cb95792` committed all 1,600,000 vertices and
4,000,000 edges with zero rejects. Its 9,619-byte migration report has SHA-256
`4816412aece55c3c70170779503c6ba0477b2ac903c6b4ea154e1a68afd678dc`.

The isolated read-only verifier regenerated the frozen P1 fixture and compared
all 5,600,000 typed records in 64 canonical ranges. Properties, identities and
relationship endpoints matched, and expected and actual canonical roots were
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
The 23,215-byte verifier report has SHA-256
`fe3f6e23e19230c8f54de8615bcbe4bf15932bacbc516ed878515c0cb2d90bff`.

The final GUI health check reported the runner idle, 5.18% disk use, zero swap
and zero boot OOM events. Target storage was 12.79%; no failed Azure activity
event was present from migration start through shutdown. Both VMs are
deallocated and the Flexible Server is Stopped. Data and raw evidence remain
retained. See the [execution sheet](az-n526-completion-20260912.md),
[redacted qualification evidence](evidence/az-n526-qualified-20260912.json) and
[credential-recovery evidence](evidence/az-n526-password-recovery-20260912.json).

### Headless source preparation — active

The operator is travelling and cannot use the installed GUI. GUI state and
SecretStorage-dependent AZ-N526 are preserved without credential reset or
substitute execution. GUI-independent source preparation and product work is
tracked in the [headless checkpoint](headless-source-preparation-20260907.md).
PostgreSQL Flexible Server, PostgreSQL-on-VM and Cosmos P1 preparation passed.
The actual AGEFreighter Linux artifact also completed all-record inventories
for AZ-PGVM, AZ-PGFS, AZ-COSMOS and the IP/port-only OP-PG simulation. Each
returned 1.6 million vertices, 4 million edges and all 18 exact label counts.
Cosmos retains only Data Reader and is back at 4,000 RU/s. The source VM is
deallocated and the Flexible Server is stopped. These are source-fixture and
source-read results, not additional guided-path migration qualifications.

### Headless checkpoint — AZ-N526 retained; four network sources read completely

AZ-N526 has completed the private target deployment, AGE preload restart and
same-VM resize from the discovery SKU to `Standard_D4s_v5`. The Neo4j 5.26.30
source and runner remain unchanged and the retained workflow is ready to start
its create-only migration. macOS is locked while the operator is travelling, so
VS Code SecretStorage cannot release the existing target credential to a
headless process. The credential was not reset or copied, no substitute job was
created, and no GUI qualification is claimed. The runner and source are
deallocated and the PostgreSQL target is Stopped; exact resume state and guest
evidence are retained.

While GUI work is unavailable, the PostgreSQL and Cosmos guided paths were
extended to support complete mapped-record inventories on the Linux runner,
per-label and capacity evidence admission, source-specific runner capabilities,
and protected PostgreSQL/Cosmos create-only migration dispatch. PostgreSQL uses
one exported repeatable-read snapshot. Cosmos requires the source-immutability
window because there is no cross-container transactional snapshot. The actual
commit-pinned Linux binary then read and decoded all 5,600,000 records for
AZ-PGVM, AZ-PGFS, AZ-COSMOS and the IP/port-only OP-PG simulation. All four
inventories passed every exact label count. These source-read results do not
qualify target deployment, migration or canonical verification; those four P1
paths remain open until their complete guided runs pass.

### AZ-N44 qualification — PASS at 12:59Z

The installed VS Code 1.136.1 GUI completed the Azure Neo4j 4.4.48 path from
resource selection and complete inventory through private target deployment,
same-VM discovery-to-migration resize, load and independent verification. Job
`45e2d8bb-641b-424d-9074-d55e14b6ac2a` committed all 1,600,000 vertices and
4,000,000 edges with zero rejects. The read-only verifier compared all
5,600,000 typed records in 64 ranges; expected and actual canonical root are
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
See the [execution sheet](az-n44-completion-20260906.md) and
[redacted evidence](evidence/az-n44-qualified-20260906.json). Both VMs are
deallocated and the Flexible Server is Stopped; data and evidence are retained.
This passes AZ-N44 only; seven other remote-source branches remain open.

### CSV-MAC qualification — PASS at 07:47Z

The installed VS Code 1.136.1 GUI completed the full local-CSV path. Job
`99f22ed1-d5b4-4432-b761-063e923b726c` committed all 1,600,000 vertices and
4,000,000 edges in 1,120 batches, with zero failed batches and zero rejects.
The hash-verified counts report passed. An isolated read-only verifier then
regenerated the frozen P1 fixture and compared typed properties, identities and
endpoints in all 64 canonical ranges. All 5,600,000 records matched; expected
and actual root are
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
See the [redacted qualification evidence](evidence/csv-mac-qualified-20260906.json).
This passes CSV-MAC only; it does not qualify the eight remote-source branches.
The runner VM is deallocated and the Flexible Server is Stopped; all resources,
CSV data, graph data, private reports and guest evidence remain retained.

### Earlier checkpoint — 07:20Z

The Mac is unlocked. The actual installed GUI has reconciled the private PG18
target, applied AGE preload, and completed the same-VM resize to D4s_v5.
VM and target are running within the unchanged USD 800 / Sep 9 08:55Z limits.
The first migration failed before graph/metadata creation; a GUI read-only
diagnostic proved the empty target, and the failed job is retained in history.
The repaired guest now reuses the established hosted AGE session initialization.

Pinned guest: `2.4.0-dev.72f45536b052`. Its new full inventory
`bcb9a509-7558-412a-879e-c80e4ae45a89` has been imported and displayed in VS Code:
all 1.6M vertices / 4M edges / 18 mappings, no errors, unchanged input hashes.
Report SHA-256 `a66e2231baf6b785917538ad68a7143deab7ecf9fec1c0bcd7c33b1cc7871af0`.
Next: approve a new create-only migration, import complete counts verification,
then run the isolated full P1 canonical verifier through the development GUI.
Extension tests: 153 pass; relevant Go tests pass. CSV-MAC and the other eight
scenario branches remain unqualified until their actual end-to-end evidence passes.
See the [execution sheet](csv-completion-20260906.md) for retained failures,
diagnostics, artifact pins and archived ARM receipts. Entries below are historical.

## Earlier inventory / target checkpoints

CSV-MAC completion is the active scope. **The actual installed VS Code GUI
completed the Linux whole-source CSV inventory on 2026-09-06**: 1.6 million
vertices + 4 million edges across all 18 mappings, no rejects, and matching
before/after file hashes. The worker ran from 03:43:29Z to 03:43:52Z (22.97 s).
See [live evidence](evidence/csv-inventory-linux-20260906.json) and the
[execution sheet](csv-completion-20260906.md). This is source inventory, **not
a migration pass** or a uniqueness/endpoint/canonical-graph proof.

After the Mac was unlocked, the GUI imported and displayed the exact 3,220-byte
whole-source report. No source scan was replayed and no private workflow file
was manually patched. The current `c880a67` guest and all old evidence are retained.

The actual GUI then reviewed source counts, live prices/capabilities/quotas,
selected a save folder and submitted a private PostgreSQL 18 target. ARM reports
**Succeeded**, server initially **Ready**, D4ds_v5 / 128 GiB / Japan East zone 1, with public
access disabled and a dedicated delegated subnet in the runner VNet. Shared
preload `pg_stat_statements,age` is pending a restart. No migration has started.
Compute quote: USD 0.736/hour for the planned D4s_v5 runner plus target; USD 100
accrued/non-compute reserve under the unchanged USD 800 / Sep 9 08:55Z deadline.
This is a conservative exposure gate, not a finalized bill.

The same-VM resize state machine and fixed Linux CSV prepare/load/counts-verify
sequence are now implemented for qualification, with a retained UUID before
target writes, explicit approvals, no automatic resume/replay, and independent
report-hash validation. They require a new pinned guest artifact and a fresh
complete inventory before use. Local extension tests pass (148); live R5 and
the full P1 property digest remain unqualified. The Mac locked again before
GUI target reconciliation. Latest guest health at 05:23:29Z was idle, disk 6%,
swap 0, kernel OOM 0. Fresh ARM reads confirm VM **deallocated** and Flexible
Server **Stopped**. All data/resources/evidence are retained; disk, storage and
network charges continue. Flexible Server can automatically start after seven
days; the authorized September 9 deadline is earlier and remains binding.
Unlock is required for the next actual GUI steps; no metadata was patched to
claim a completed deployment reconciliation or migration.

## Earlier retained transport and sample checkpoints

Latest checkpoint: **the bounded Linux CSV trial passed its transport and
assessment controls**, not a P1 migration. After user reauthentication, the
installed VS Code 1.136.1 GUI verified all **18 CSV imports / 1,168,576,671 bytes**.
Independent Linux full-byte readback matched every desktop manifest. All 18
typed mappings matched the P1 reference. A GUI-approved Linux sample profile
completed and its **26,514-byte** report was hash-verified, retained privately
and displayed in VS Code. See [Linux trial evidence](linux-runner-20260905.md)
and [full-file readback](evidence/csv-guest-readback-20260906.json).

The profile itself reports **incomplete**, as designed: it observed only a
bounded prefix of 10,000 vertices / 0 edges. Mapping validation and read-only
checks passed; whole-source counts, capacity acceptance and migration success
remain unproven. No target or migration has started. VM deallocation was
requested at 02:18Z and independently confirmed; disk, CSVs, reports and failure
evidence are retained.

Corrected the earlier blocker attribution: old MFA errors did not prove the
cause of every later PUT failure. Fresh diagnostics exposed HTTP 400 with
exactly Azure's maximum 25 Managed Run Command resources. Safe capacity
preflight and bounded HTTP diagnostics were added; 127 unit tests, package and
CI `34004772287` passed. Completed command receipts were archived/pushed and
freshly hash-matched before removing only historical successful ARM command
resources. Latest and failed receipts and all guest data were retained. This
was operator maintenance, not automatic housekeeping or a source replay.

## Authorization / live resources

- Subscription: `MCAPS-Hybrid-REQ-51508-2023-rifujita` (selected account verified).
- Approved ceiling: 800 USD; live window: 96 hours from first creation.
- Conservative first creation: **2026-09-05T08:55:00Z**; deadline:
  **2026-09-09T08:55:00Z**. No old P3 authorization or result is substituted.
- Created dedicated `rg-af-vscode-p1-20260905-a` in Japan East and its tagged
  `vnet-af-vscode-p1` / `runner` compute subnet. Both provisioning states succeeded.
  The first one-VM Linux trial was submitted through the installed GUI at
  **2026-09-05T12:48:08Z**, after the user approved the pinned development build.
  At that initial checkpoint no Flexible Server or Cosmos resources existed;
  the CSV Flexible Server described above was subsequently created. Private run metadata
  retains exact IDs; ARM completion is not a guest or migration pass.
  Initial reserve was **25 USD**, superseded by the **100 USD** target-plan reserve above. B2s_v2 Linux compute is 0.109 USD/hour;
  Standard NAT is 0.045 USD/hour plus 0.045 USD/GB, and its Standard public IP
  is 0.005 USD/hour. At most 92 remaining hours plus a 10 USD disk/data reserve
  fits the initial ceiling. The global 800 USD / 96-hour limits are unchanged.
  A bounded pre-creation setup watcher installs a 16:00 UTC Azure daily shutdown
  on the exact owned VM once it exists, or deallocates it if setup fails.
- Corrected an initial preflight interpretation: `defaultOutboundAccess=false`
  **disables** implicit egress ([Azure documentation](https://learn.microsoft.com/en-us/azure/virtual-network/ip-services/default-outbound-access)).
  Created zone-1 Standard NAT `nat-af-vscode-p1-egress` and its public IP on the
  task's runner subnet; no VM public IP, ingress rule, peering or source firewall
  change was made. No exception tag was applied to either network resource.
  NAT/IP incur charges even with the VM stopped; retain them within the same
  deadline/cost gate until later qualification or separately authorized cleanup.
- Actual installed VS Code 1.136.1 GUI selected CSV/local, enumerated the existing
  Azure Resources account, and selected the authorized subscription and dedicated
  resource group. This is account/placement evidence, not an Azure migration pass.
- After action-time approval, the installed GUI created Standard LRS account
  `af83c6b829acdc4405aa2dfb`, its workflow container and operator Storage Blob
  Data Contributor scoped to that account only. Deployment succeeded at
  **2026-09-05T09:24:00Z**. Anonymous and shared-key access remain disabled.
- **Initial network blocker (resolved for this account):** the enforced `StorageAccount_PublicNetwork_Modify`
  policy in `MCAPSGovDeployPolicies` modified `publicNetworkAccess` to `Disabled`.
  The retained activity observation identifies the modification at 09:23:33Z;
  a read-only authenticated Blob request failed the storage network rules.
  This was not an Azure sign-in failure. The policy definition was subsequently
  read: its explicit exclusion defaults are `SecurityControl=Ignore` on a resource
  or RG. The user identified this as the official development exception and
  explicitly authorized it **on this storage account only**. Assignment parameter
  reads were denied, so effective behavior was verified rather than assumed.
- Merged the approved tag on the owned account, preserving ownership tags, then
  enabled its authenticated public endpoint. The setting remained `Enabled`;
  anonymous access and shared keys remained disabled. The RG has no exception tag.
  An actual `Carrier.csv` upload at 10:00:18Z and authenticated download passed:
  235,855 bytes, SHA-256
  `0a2c6e8ecf1fdfe2540c4058202527e458f66e1d7611d75c03d51144089fe88b`.
  This single-file CLI probe is **not** a GUI/P1 migration pass.
- The installed GUI selected all 18 CSV files (1,168,576,671 bytes). Its first
  upload failed before committing a blob. Inspection found that azureauth's
  `AzureSubscription.credential.getToken` returns a captured ARM token regardless
  of requested scope. File/report/archive transfers now request a Storage-scoped
  session for the same VS Code account, without ARM/shared-key/account fallback.
  Regression tests cover scopes, refresh and missing/foreign sessions. The GUI
  retry uploaded **all 18 files / 1,168,576,671 bytes**, and independent full-byte
  authenticated readback matched every size and SHA-256 at **10:17:36Z**.
  See [retained transfer evidence](csv-transfer-20260905.json). Anonymous read was separately rejected
  (HTTP 409), while the signed-in user's readback succeeded.
- Source servers were not prepared; preparing dedicated sources is authorized.

## Completed local foundation

1. Saved and reviewed the [implementation and qualification plan](plan.md).
2. Added opt-in typed CSV properties for scalar/array integers, floats, booleans
   and strings. Legacy mappings remain strings; type changes invalidate resume.
3. Added `verify --require-complete`, preserving report output while failing
   incomplete verification. Existing default CLI behavior remains compatible.
4. Added a controller-side counts-verification decision module with wrong-job,
   stale evidence, digest mismatch, reject, missing coverage and count mismatch
   tests. **It is not yet connected to an enabled guided migration operation.**
5. Regenerated full P1 on MacStudio: 1,600,000 vertices + 4,000,000 edges,
   64 shards, seed 20260829. All 1,170 fixture files match the earlier fixture root:
   `f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f`.
6. Exported 18 headered CSV files, 18 typed Cosmos-ready JSONL files, mapping
   metadata and checksums. All 5,600,000 JSONL documents are now present in the
   dedicated private Cosmos account; the exact remote-count seal is tracked in
   the headless checkpoint above.
7. Read **all 5,600,000 converted CSV records** through AGEFreighter's actual CSV
   connector and compared all 64 canonical ranges with the original fixture.
   Both roots are:
   `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
8. Added the [R3 Linux assessment boundary](runner-assessment-protocol.md):
   protected control transport, boot-bound readiness, durable read-only workers,
   no automatic replay/reboot resume, and hash-verified bounded report retrieval.
   Actual CSV profiling runs through the child-process boundary in tests. The
   GUI can request guest readiness independently of deployment state.
9. Added field-based source forms for Neo4j, PostgreSQL, Cosmos explicit/Gremlin
   documents and CSV. They generate configurations without YAML or SQL input;
   CSV includes property types and an explicit null marker. Local drafts can be
   reviewed before a release is available or any VM is created.
10. Wired approved network-source profiles and complete Neo4j, PostgreSQL and
    Cosmos inventories to native secret prompts, protected dispatch and retained
    status checks. Successful operation manifests remain in history. Finished
    workers are not accepted as capacity or migration passes. PostgreSQL/Cosmos
    whole-source inventory and migration dispatch are locally tested but not
    Azure-qualified.
11. Implemented guest/controller bulk-report export/import with a single bounded
    data transfer, independent full hash/size verification, private immutable
    local retention and loss-of-acknowledgement reconciliation. Exact storage
    ownership and non-anonymous/shared-key-disabled policies are checked before
    transfer. The GUI now connects reviewed Standard LRS storage creation,
    account-scoped user Blob Data Contributor, SDK user-delegation capability
    issuance through the existing Azure login, and a script-disabled report viewer.
    The designed endpoint is network-public/authenticated HTTPS, not
    private-endpoint network isolation; this is disclosed before approval. Live
    provisioning subsequently exposed the policy blocker recorded above.
12. Connected streamed CSV review/upload (8 MiB blocks, conditional commit,
    content-addressed destinations) and protected asynchronous guest import.
    Guest full size/hash seals, 80% disk gate, 10-minute download limit and
    non-replaying workers are required before CSV profiling. Limits are 2 GiB per
    file / 10 GiB per workflow. Failed/interrupted guest import repair and bulk
    automatic import orchestration are still open; imports are approved per file.
13. Added explicit user-level opt-in for reviewed commit/hash-pinned Linux test
    artifacts in workflow storage. Production still requires an official release.
    VM identity gets container-scoped Blob Reader only; cloud-init fetches through
    managed identity with no SAS/token in the template. Guest readiness also
    checks the pinned commit. The build helper requires a clean committed tree;
    no mutable-branch guest builds or unapproved release publication are used.
14. Actual installed-GUI preparation exposed and corrected the ARM what-if
    response envelope and existing-resource `Ignore` handling. Only independently
    enumerated unchanged resources may be ignored; unexpected mutations still
    fail review. Restored drafts now restore source/placement fields. Added
    shallow, bounded CSV-folder selection and explicit storage-network status
    with a pre-transfer stop for disabled/perimeter-restricted public access.
15. The Linux trial GUI exposed a real Retail Prices ambiguity: Bsv2 Cloud
    Services shares `serviceName=Virtual Machines` and the ARM SKU with ordinary
    Linux compute. Excluded that distinct product without selecting arbitrarily
    among ambiguous Linux meters. All **117 unit tests**, typecheck, packaging,
    and CI **33966950563** passed on `050ee1f`; installed bundle hash is
    `31d154d7bb2d00b0ee87b6d26f4b1db2078e54a66123c30774cf374f84296b72`.
    The GUI then displayed the unique 0.109 USD/hour price and passed what-if.
    The guest build remains the separately approved `6ff072c0db71` archive,
    SHA-256 `5fee11ec3fde605ef7e18afaa56131862113348de8ddcb8bc37b52aba1c3620b`;
    no release/Marketplace publication was made.

Local retained data (ignored by Git):

- `production-simulation/work/vscode-p1-20260905/manifest.json`
- `production-simulation/work/vscode-p1-portable-20260905/portable-manifest.json`
- `production-simulation/work/vscode-p1-portable-20260905/csv-source.json`
- `production-simulation/work/vscode-p1-portable-20260905/canonical-verification.json`

## Required execution stages still open

Validation of this foundation: `go test ./...` passed; extension typechecking and
all 110 unit tests passed at the transfer/CSV/development-artifact checkpoint. Targeted guest/tools
tests also passed with Go's race detector. All five GUI-generated source formats passed the actual
Go CLI validator; the contract test is now part of extension CI. These commands
do not run the Azure P1 GUI scenarios.
At the preceding checkpoint, nine Linux guest tests passed under Apple Container/Rosetta,
including validation of the generated unit by `systemd-analyze verify`. This
does not exercise a running service manager or an Azure VM agent. The installed
VS Code 1.136.1 executable passed three isolated Extension Host smoke tests,
including opening all four source editors without a release, workspace, CLI or
ARM calls. The actual webview scripts also have source-branch and edit/approval
gating tests. This is not a live connected GUI migration.

The first branch CI run exposed an existing Windows-specific permission test:
POSIX group/other bits cannot verify a Windows ACL. Runner storage now explicitly
sets a current-user-only inheritable Windows ACL and its Windows test checks the
actual file ACL. A failed ACL setup blocks reads/writes; macOS/Linux retain
owner-only modes. CI also uses the extension's actual VSIX version and supported
minimum VS Code 1.105.0 instead of the stale 2.3.0/1.100.0 constants.
The follow-up [CI run 33952182561](https://github.com/rioriost/agefreighter/actions/runs/33952182561)
on commit `75f7885d2d4d31bd778870be3866191d027730c6` passed all six jobs:
90 unit tests on each of Linux, Windows and macOS, the five actual-CLI source
contracts, the Extension Host suite, and VSIX packaging. In particular, the
Windows private-directory and inherited-file ACL checks passed on Windows.

At the bulk-report checkpoint, [CI run 33954366675](https://github.com/rioriost/agefreighter/actions/runs/33954366675)
on commit `67ff793b8e00efd2096e290b1541a37461d81f9d` passed all six jobs.
Each of Linux, Windows and macOS passed 99 extension tests. The Linux job also
ran guest/tools tests with the race detector and validated all five source
configuration formats through the real CLI. Extension Host and packaging passed.
MacStudio's actual VS Code 1.136.1 separately passed all three isolated host
smoke tests after this change, and the matching VSIX was installed. None of
these tests exercised Azure storage provisioning, real SAS/RBAC, or P1 migration.

| Stage | Current status |
|---|---|
| Dedicated Azure fixture topology / ownership and cost watchdog | RG/VNet/subnet, explicit NAT, transfer storage/RBAC and one private runner VM tested; account-only approved exception; exact-VM 16:00 UTC shutdown enabled; whole-suite cost automation remains open |
| Source preparation: Neo4j 4.4 / 5.26, PG VM / FS, Cosmos | All dedicated source fixtures are retained. PGVM r7, PGFS r3 and Cosmos r3 passed exact preparation checks; all six VMs are deallocated and all four Flexible Servers are stopped |
| P1 local CSV | Prepared; complete local canonical comparison passed; installed-GUI storage upload, Linux import/sealing and independent full-byte readback all passed for 18 files |
| R3 remote source configuration, mapping, assessment, upload | CSV-MAC and Azure Neo4j 4.4 passed in the installed GUI. Commit-pinned headless complete inventories also passed for AZ-PGVM, AZ-PGFS, AZ-COSMOS and OP-PG; this does not promote them to full guided-path qualifications |
| R4 target deployment and same-VM resize | CSV-MAC and AZ-N44 actual Azure paths passed; remaining source branches open |
| R5 durable migration / resume / verification controller | CSV-MAC and AZ-N44 clean migrations, counts and canonical verification passed; recovery and remaining source branches open |
| Installed VS Code 1.136.1 full GUI branches | CSV-MAC and AZ-N44 passed; seven branches not run |
| Nine P1 base paths and additional branch/failure ledger | 2 / 9 complete |

The current installed preview has two end-to-end GUI/Azure qualifications
(CSV-MAC and AZ-N44). It must not be described as qualified for the other seven branches.
The preview VSIX is installed into MacStudio's VS Code 1.136.1. Installation
and bundle identity are rechecked with each packaged update; these do not imply
that the live GUI branches passed.
After the Storage-scoped session fix, typechecking, all 116 unit tests and packaging
passed. The installed and built JavaScript SHA-256 is
`92d50336ae387cb1be19c81fcc60551f948ede95085943bc393c32840622c7a7`.
An already-open extension host needs a window reload to pick up this build.
No Marketplace publication was performed here.

At that transfer checkpoint, [CI run 33956445191](https://github.com/rioriost/agefreighter/actions/runs/33956445191)
passed all six jobs: 110 unit tests on Linux/macOS/Windows, five real-CLI source
contracts, Extension Host, and packaging. The exact Linux development archive is
retained locally with its build manifest; it has not been uploaded or executed.

The subsequent [CI run 33957782167](https://github.com/rioriost/agefreighter/actions/runs/33957782167)
for `4c12b32756a7e405a64f0d186a9dbf7fdbee45b7` (what-if and restored fields)
also completed successfully. This predates the final network guard/folder tests.

Storage-audience fix [CI run 33960028041](https://github.com/rioriost/agefreighter/actions/runs/33960028041)
on `c9d477b5003189082a575ab7afd7a67b1780cdbb` passed all six jobs: Linux/macOS/
Windows unit tests, source contracts, Extension Host and packaging.

## Review notes for the next implementation stage

- Matching release/bootstrap is still mandatory in production. The released 2.4
  artifact remains a production gate. The approved test-only commit/hash-pinned
  artifact and managed-identity bootstrap passed the actual Azure Linux CSV trial.
  Do not install a mutable branch on guests or publish an unapproved release.
- Managed Run Command capacity now fails before submission at 25 resources.
  Evidence-preserving lifecycle management is still manual. Plan bounded command
  retention before long GUI sessions; do not delete active/uncertain commands or
  guest data to recover capacity.
- The live CSV profile sampled 10,000 vertices and no edges. Do not feed its
  lower-bound estimates into automatic VM/target sizing. Whole-source inventory
  or explicitly representative assessment remains an R4 acceptance gate.
  Complete CSV inventory has now passed local P1 testing, and prefix-scaled
  estimates no longer pass the capacity gate even with exact total counts.
- The form requires explicit reviewed mappings; automatic PostgreSQL schema/FK
  recommendations are not implemented. Current table/column/graph identifiers
  are limited to ASCII letters/digits/underscores. Cosmos explicit mappings use
  a top-level label field; only the public Azure NoSQL endpoint is supported.
- TLS validation remains mandatory. A custom Neo4j/PostgreSQL source CA can now
  be selected locally, digest-bound to the reviewed source, rechecked at each
  approved operation, transported only as a protected parameter, validated as
  CA-only PEM on Linux, and removed after use. Live private-IP qualification is
  still required; hostname verification is never disabled.
- Neo4j inventory uses count-store totals; generic profile `exact` is still
  bounded to 1,000,000 rows and must not masquerade as a full inventory.
  PostgreSQL, Cosmos and CSV now have connector-specific, explicitly approved
  complete streams with row/time bounds (and Cosmos RU disclosure). Their local
  implementation does not replace path-specific Azure migration qualification.
- Managed Run Command instance-view output is limited to 4 KB. Use it only for
  bounded control/acknowledgements, not full reports or CSV transfer. Source
  secrets require protected parameters; raw command output is not safe UI data.
  [Azure managed Run Command](https://learn.microsoft.com/ja-jp/azure/virtual-machines/linux/run-command-managed).
- Keep exact-count verification and full canonical property/endpoint equality
  separate. A count pass alone cannot qualify the P1 scenario.
