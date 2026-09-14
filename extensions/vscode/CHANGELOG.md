# Changelog

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
