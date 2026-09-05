# Guided migration P1 qualification progress

Updated: 2026-09-05. Overall outcome: **not yet qualified**.

## Authorization / live resources

- Subscription: `MCAPS-Hybrid-REQ-51508-2023-rifujita` (selected account verified).
- Approved ceiling: 800 USD; live window: 96 hours from first creation.
- Conservative first creation: **2026-09-05T08:55:00Z**; deadline:
  **2026-09-09T08:55:00Z**. No old P3 authorization or result is substituted.
- Created dedicated `rg-af-vscode-p1-20260905-a` in Japan East and its tagged
  `vnet-af-vscode-p1` / `runner` compute subnet. Both provisioning states succeeded.
  No compute, disks, Flexible Server or Cosmos resources have been created.
  Compute remains gated on the cost/deadline watchdog and
  explicit outbound/network review. Private run metadata retains exact IDs.
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
  See [retained transfer evidence](csv-transfer-20260905.json). No development
  archive or guest execution has occurred. Anonymous read was separately rejected
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
   metadata and checksums. These JSONL files are not yet imported into Cosmos.
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
10. Wired approved network-source profile and Neo4j inventory to native secret
    prompts, protected dispatch and retained status checks. Successful operation
    manifests remain in history. Finished workers are not accepted as capacity
    or migration passes. CSV dispatch is blocked pending verified upload.
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
| Dedicated Azure fixture topology / ownership and cost watchdog | RG/VNet/subnet and transfer storage/RBAC created; account-only approved exception and CSV probe passed; compute/cost watchdog not yet enabled |
| Source preparation: Neo4j 4.4 / 5.26, PG VM / FS, Cosmos | Not run |
| P1 local CSV | Prepared; complete canonical comparison passed; installed-GUI upload and all 18 remote file readback hashes passed |
| R3 remote source configuration, mapping, assessment, upload | Forms, mappings, approved start/status, storage/RBAC/report GUI, CSV upload/seal and pinned test artifact implemented locally; real Azure qualification, schema suggestions and complete assessment evidence remain open |
| R4 target deployment and same-VM resize | Implementation required |
| R5 durable migration / resume / verification controller | Implementation required |
| Installed VS Code 1.136.1 full GUI branches | Not run |
| Nine P1 base paths and additional branch/failure ledger | 0 / 9 complete |

The current installed preview must not be described as an end-to-end migration
product. The new local tests and fixture digest are not GUI/Azure qualifications.
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
  artifact is unavailable (rechecked). Test-only commit/hash-pinned artifact upload
  and managed-identity bootstrap are implemented but await real Azure validation.
  Do not install a mutable branch on guests or publish an unapproved release.
- The form requires explicit reviewed mappings; automatic PostgreSQL schema/FK
  recommendations are not implemented. Current table/column/graph identifiers
  are limited to ASCII letters/digits/underscores. Cosmos explicit mappings use
  a top-level label field; only the public Azure NoSQL endpoint is supported.
- TLS validation is mandatory, but custom source-CA upload/installation is not
  yet implemented. Resolve this before private-IP fixture qualification; never
  silently disable certificate validation.
- Neo4j inventory uses count-store totals; generic profile `exact` is still
  bounded to 1,000,000 rows and must not masquerade as a full P1 inventory for
  PostgreSQL/Cosmos/CSV. Implement connector-specific count evidence or explicitly
  approved complete scans with bounds and RU costs before automatic sizing.
- Managed Run Command instance-view output is limited to 4 KB. Use it only for
  bounded control/acknowledgements, not full reports or CSV transfer. Source
  secrets require protected parameters; raw command output is not safe UI data.
  [Azure managed Run Command](https://learn.microsoft.com/ja-jp/azure/virtual-machines/linux/run-command-managed).
- Keep exact-count verification and full canonical property/endpoint equality
  separate. A count pass alone cannot qualify the P1 scenario.
