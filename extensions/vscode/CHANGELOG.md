# Changelog

## 2.4.0

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
