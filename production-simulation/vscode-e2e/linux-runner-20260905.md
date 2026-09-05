# First Linux runner qualification — 2026-09-05

Status: in progress; not a P1 migration qualification.

## Deployment and installation

- Actual Mac VS Code 1.136.1 GUI, existing Azure Resources login, saved workflow
  `83c6b829-acdc-4405-aa2d-fb2f2d99af9f`. No desktop AGEFreighter invocation is
  used by the guided workflow.
- Dedicated Japan East / zone 1 B2s_v2 (2 vCPU, 8 GiB), Ubuntu 22.04 Gen2,
  64 GiB Standard SSD. ARM submitted 12:48:08Z; GUI reconciled provisioned
  12:49:07Z. No VM public IP or SSH ingress.
- Explicit Standard NAT egress on the dedicated subnet, which retains
  `defaultOutboundAccess=false`. Storage policy exception remains on the transfer
  account only. VM identity Blob Reader is scoped to the workflow container.
- Azure automatic shutdown independently verified `Enabled`, 16:00 UTC, exact
  owned VM target. No automatic start is configured.
- Guest readiness initiated through the GUI at 12:49:29Z and reconciled:
  Linux amd64, `2.4.0-dev.6ff072c0db71`, commit
  `6ff072c0db71a3ae1e42a72e3c336a04ac284df8`, archive SHA-256
  `5fee11ec3fde605ef7e18afaa56131862113348de8ddcb8bc37b52aba1c3620b`.
- Independent guest health at 12:51:30Z: cloud-init done; disk 4% used
  (2,316,439,552 / 66,404,147,200 bytes); memory available 7,734,779,904 bytes;
  swap configured/used zero; `pswpin`, `pswpout`, `oom_kill` all zero.
  Private runner directory is root-owned, mode 0700.
- Installed executable SHA-256:
  - agefreighter: `c99b483b9badbd1a784b93a034991f7c983cd90411b7b21314195ca1253fcd84`
  - agefreighter-tools: `cf8e2f3259619e443217bc9229bd1a8009bbf7084fa1806e7b6b1f005bdf1878`

## Live defects found and corrected

1. Cloud Services pricing shares the ordinary Linux VM SKU and service name.
   The parser now excludes that different product, while retaining ambiguity
   rejection for ordinary Linux meters. `050ee1f`, 117 unit tests and CI
   `33966950563` passed. The real GUI then accepted 0.109 USD/hour.
2. The first `CARRIED_BY.csv` import at 12:50:46Z sent the controller's `phase`
   field as part of a structurally compatible manifest. Go correctly rejected
   unknown fields (`invalid runner request`, exit 1). It did not execute a
   source read: an independent guest check proved the operation directory and
   CSV destination absent. Failed command and operation identities are retained.
   `26495dd` sends only the explicit file/size/hash/URL wire fields. A GET-only
   reconciliation of the exact pre-execution rejection retains the failed
   attempt and makes the file eligible for another separately approved import;
   ambiguous or post-execution failures are never reset/replayed this way.
   All 120 unit tests, packaging and all six CI jobs (`33967456422`) passed.
   Installed extension bundle SHA-256:
    `e430a70648c5ba5a388280197ec30cbc83ef1382fea8ffdf82b7d43b10b23baa`.
3. Additional macOS runner regression tests exposed the test fixture's
   `/var` versus `/private/var` temporary-directory alias. Canonicalized only
   the fixture root, preserving production symlink-escape protection.
   `go test ./internal/runner` subsequently passed on this Mac.

## GUI mapping review

Entered all nine vertex and nine edge mappings through the actual source-form
fields. Independently compared the persisted generated configuration against
`csv-source.json`: all labels, selected-file identities, ID columns, endpoint
labels/columns, property names and scalar/array types match exactly. This is
mapping evidence, not a source profile or successful migration.

## Boundaries

The 18 P1 CSV files already passed actual GUI upload and independent full
readback on the desktop. This trial separately tests guest import/sealing and
bounded source assessment. Neither installation, ARM success nor a source
profile proves successful migration. All nine full migration cases remain open.
The task's 800 USD / 96-hour window ends 2026-09-09T08:55:00Z. This one-VM
trial uses a 25 USD reserve and must stop before its 16:00 UTC daily safeguard.
