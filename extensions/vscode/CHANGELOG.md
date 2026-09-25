# Changelog

## 2.4.0

- Add the runner-first guided workflow for CSV, Neo4j, PostgreSQL and Cosmos DB
  for NoSQL: reviewed source mappings, protected credentials, complete inventory,
  private PostgreSQL 18/AGE target planning, same-VM resize, migration, explicit
  same-job recovery and hash-verified report import. Advanced LoadJob commands
  remain available through the local CLI.
- Add PostgreSQL catalog and reviewed FK mapping suggestions, private-CA
  transport, typed CSV/Cosmos properties and connector capability gates.
- Preserve durable operation/job identity across reconnects, refuse active-job
  resize and incompatible layouts, and reconcile uncertain submissions without
  automatic replay. Review and archive eligible historical readiness controls
  before explicitly removing them.
- Complete the defined installed-GUI P1 base matrix (9/9) and finite extended
  branches (12/12), including the final Cosmos managed-identity access and active
  inventory resize-refusal cases. Full canonical migration verification remains
  separate from inventory/counts; each branch retains its stated evidence layer
  and limitations. See the [qualification ledger](../../production-simulation/vscode-e2e/remaining-validation.md).
- Guided targets remain private PostgreSQL 18/AGE, single-server/HA-off and
  create-only. Native Cosmos Gremlin API, automatic network exposure, general
  command retirement and automatic compute shutdown/cleanup are not added.
  M6 packaging/compatibility and actual release publication remain separate
  from the completed qualification matrix.

### Historical development checkpoints

The entries below preserve their observations at the time they were written.
Their pending, disabled and unqualified labels describe intermediate builds,
not the current feature status summarized above.

- Add optional Cosmos `propertyTypes` through `name=field:type` mappings. Preserve
  declared floats even when JSON spells them as integers, bind types to resume
  fingerprints, and block typed jobs on runners without the new capability.
  Existing untyped jobs retain their inference; changed mappings require fresh
  jobs. Full P1 offline parity passes; Azure Cosmos GUI requalification is pending.

- Reject PostgreSQL assessment/migration on older Linux runners that do not advertise native SQL float preservation; retain read-only access to failed-run evidence.

- Incorporate the 2.3.1 PostgreSQL floating-point preservation and gRPC security
  fixes without downgrading the runner-first guided UI or its dependencies.

- Add complete CSV inventory, a data-preserving pinned Linux upgrade, and private
  CSV target review/export with live service, quota, network and price gates.
  Target approval uses native fields and SecretStorage, create-only what-if and
  GET-only reconciliation. Add an explicitly approved, evidence-preserving same-VM
  resize and a pinned Linux CSV prepare/load/complete-counts verification preview.
  Retain job UUID before writes; require fresh idle/disk/swap/OOM health, verified
  TLS, complete re-inventory after upgrade, and no automatic resume or replay.
  Live migration and independent property-digest qualification remain open.
  The isolated P1 Linux GUI inventory counted all 5.6M rows; this is not a
  migration qualification or a public release of the completed guided workflow.
- Clear an earlier sample report's imported indicator when a new inventory starts.

- Request Storage-scoped sessions for the selected existing VS Code account for
  CSV/archive uploads and report capabilities. The subscription SDK credential
  is ARM-only and ignores requested scopes. Fail missing/foreign sessions without
  account/key fallback, and include the required UTC date on upload requests.

- Reconcile the ARM what-if response envelope and independently known unchanged
  resources; restore saved source/placement fields. Add bounded CSV-folder
  selection. Display deployed storage network state separately from provisioning
  and stop the current desktop transfer path when governance disables public
  networking, without changing policy or suggesting another sign-in.

- Wire approved transfer storage/RBAC, existing-account delegation and full report
  import into the source GUI. Add streamed CSV block upload, explicit asynchronous
  guest import with full-hash seals, disk gate and no replay. Add user-opt-in
  commit/hash-pinned test artifacts with container-scoped managed-identity reads.
  These paths are locally tested, not yet Azure P1 qualified; R4/R5 remain open.

- Add an immutable bulk-report protocol and private local import: exact report
  identity/size/hash, short-lived single-blob user-delegation capabilities,
  conditional create-only export, GET-only recovery after uncertain responses,
  original int64/Unicode JSON bytes and no-replace evidence retention. Validate
  owned HTTPS/non-anonymous/shared-key-disabled storage before transfer. This
  protocol foundation by itself is not an Azure P1 pass or a migration feature.

- Add local pre-deployment source drafts and field-based Neo4j/PostgreSQL/Cosmos/
  CSV configuration. Generate read-only table queries, Cosmos explicit/Gremlin
  mappings and typed CSV mappings with a null marker. Add reviewed native-secret
  assessment start, retained status reconciliation and successful-operation
  history. R4/R5 remain unavailable.

- Add protected Linux guest readiness controls and the local R3 assessment
  execution boundary: durable operation IDs, explicit no-replay behavior,
  boot-bound readiness, private credentials, bounded reports and hash-checked
  diagnostic chunks. This foundation is not an Azure P1 qualification.

- Replace free-text runner RG/region fields with subscription-backed dropdowns.
  Select a shared migration resource group for the runner and the future Flexible
  Server target; keep source RG and network placement independent. Support list
  refresh, source-region defaults and stale-selection rejection.

- Replace the local Neo4j wizard with a runner-first preview: choose one of four
  sources, defer workspace selection, discover ARM source candidates and review
  a private Linux discovery/migration VM. No desktop CLI is used by this path.
- Add approval-gated runner deployment with pinned release checksums, quota and
  zone checks, create-only what-if, atomic workflow records, cross-window locks,
  and status reconciliation after ambiguous responses. Target deployment,
  same-VM resize and migration remain disabled pending R4–R5.
- The earlier local profiling milestones below are retained as implementation
  history, not the current wizard flow.

- Update chat help, AI workflow guidance, the welcome view, and the installed
  guide to start with the source form. Clearly show that this development build
  ends at the Azure proposal; documentation opens the bundled version's guide.
- Add the first guided migration milestone: VS Code Azure-session discovery,
  Neo4j source connection form, protected credential materialization, verified
  Azure source placement, draft generation, and bounded source sizing evidence.
- Add a read-only Neo4j transactional count-store inventory command so bounded
  record-size observations can be scaled against exact node and edge totals.
- Add runtime PostgreSQL 18 capability, zonal Compute SKU, quota, and USD retail
  checks, then save an expiring private-network Azure deployment proposal.
- Treat the Azure account already signed into VS Code as a prerequisite; the
  guided workflow never starts a second Azure login.
- Preserve the existing LoadJob-first commands and AI read-only boundary.

## 2.3.1

- Package the stable extension alongside the CLI PostgreSQL floating-point
  correctness patch; no guided-migration development features are included.
- Retain the previously merged serialize-javascript security update.
- Recommend CLI 2.3.1; PostgreSQL checkpoints from older CLI versions require
  a new migration rather than an in-place resume with the new CLI.

## 2.3.0

- Add workspace discovery and a migration-job tree for AGEFreighter LoadJobs.
- Add guided validate, plan, profile, doctor, load, resume, status, verify,
  report, optimize, and cleanup commands.
- Keep long-running and mutating operations visible in a confirmed terminal.
- Add script-free, escaped JSON report views with bounded process capture.
- Add the optional `@agefreighter` chat participant.
- Add a confirmed, read-only language-model tool with workspace-path validation
  and recursive evidence redaction.
- Document that Windows AGEFreighter 2.3.0 CLI binaries remain unsigned.
