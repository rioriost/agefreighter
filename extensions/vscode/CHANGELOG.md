# Changelog

## 2.4.0

- Add protected Linux guest readiness controls and the local R3 assessment
  execution boundary: durable operation IDs, explicit no-replay behavior,
  boot-bound readiness, private credentials, bounded reports and hash-checked
  diagnostic chunks. Source assessment/mapping and R4/R5 remain disabled; this
  foundation is not an Azure P1 qualification.

- Replace free-text runner RG/region fields with subscription-backed dropdowns.
  Select a shared migration resource group for the runner and the future Flexible
  Server target; keep source RG and network placement independent. Support list
  refresh, source-region defaults and stale-selection rejection.

- Replace the local Neo4j wizard with a runner-first preview: choose one of four
  sources, defer workspace selection, discover ARM source candidates and review
  a private Linux discovery/migration VM. No desktop CLI is used by this path.
- Add approval-gated runner deployment with pinned release checksums, quota and
  zone checks, create-only what-if, atomic workflow records, cross-window locks,
  and status reconciliation after ambiguous responses. Remote assessment,
  target deployment, same-VM resize and migration remain disabled pending R3–R5.
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
