# Guided migration P1 qualification progress

Updated: 2026-09-05. Overall outcome: **not yet qualified**.

## Authorization / live resources

- Subscription: `MCAPS-Hybrid-REQ-51508-2023-rifujita` (selected account verified).
- Approved ceiling: 800 USD; live window: 96 hours from first creation.
- First creation / deadline: **not started**. No test Azure resource has been
  created by this work. No old P3 authorization or result is substituted.
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
    transfer. It remains off in the GUI until reviewed storage/network/RBAC and
    existing-account capability issuance are connected. No storage was deployed.

Local retained data (ignored by Git):

- `production-simulation/work/vscode-p1-20260905/manifest.json`
- `production-simulation/work/vscode-p1-portable-20260905/portable-manifest.json`
- `production-simulation/work/vscode-p1-portable-20260905/csv-source.json`
- `production-simulation/work/vscode-p1-portable-20260905/canonical-verification.json`

## Required execution stages still open

Validation of this foundation: `go test ./...` passed; extension typechecking and
all 99 unit tests passed at the bulk-report checkpoint. Targeted guest/tools
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

| Stage | Current status |
|---|---|
| Dedicated Azure fixture topology / ownership and cost watchdog | Not deployed |
| Source preparation: Neo4j 4.4 / 5.26, PG VM / FS, Cosmos | Not run |
| P1 local CSV | Prepared; complete canonical comparison passed |
| R3 remote source configuration, mapping, assessment, upload | Forms, mappings, approved start/status and bulk-report protocol implemented; report storage/capability/GUI connection, CSV upload, schema suggestions and complete assessment evidence remain open |
| R4 target deployment and same-VM resize | Implementation required |
| R5 durable migration / resume / verification controller | Implementation required |
| Installed VS Code 1.136.1 full GUI branches | Not run |
| Nine P1 base paths and additional branch/failure ledger | 0 / 9 complete |

The current installed preview must not be described as an end-to-end migration
product. The new local tests and fixture digest are not GUI/Azure qualifications.
The updated preview VSIX was installed into MacStudio's VS Code 1.136.1. The
installed and built JavaScript bundles have matching SHA-256:
`f86998718c9d60f6f419e9bdca0f3db691d23b9102457f580d0b65350d8b8b63`.
An already-open extension host needs a window reload to pick up this build.
No Marketplace publication was performed here.

## Review notes for the next implementation stage

- Matching release/bootstrap is still mandatory. The released 2.4 artifact is
  not available; introduce the explicitly reviewed commit/hash-pinned development
  artifact path before live testing. Do not install a mutable branch on guests.
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
