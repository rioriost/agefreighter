# OP-PG r1 installed-GUI qualification

Status: complete inventory passed and was hash-verified/imported; a fresh
private target is being provisioned. Migration and canonical verification
are not complete. Overall route coverage remains 4/9.

## Route boundary

- Workflow: `53625ae3-b155-4821-bfc3-910cc8cad6df`.
- Source selection: PostgreSQL / on-premises, using IP, port, database and
  read-only credentials. No source ARM ID or source Azure discovery.
- The dedicated laboratory reuses the frozen PostgreSQL 18 P1 fixture.
  Laboratory VM power/health checks are not input to source discovery.
- All 18 mappings exactly match `fixtures/postgresql-p1-mappings.json`.
- Configuration SHA-256:
  `284397bc7549f79520b426ed66bf182dba52e59e1ffb52ca457ae5691f63ada5`.
- Source CA SHA-256:
  `0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
- Linux version `2.4.0-dev.9ef16968363b`, commit
  `9ef16968363b31214324f392553f7c8e88150272`, archive SHA-256
  `10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
  This is the same reviewed native-float-preserving artifact used by AZ-PGVM r3.

## Complete source inventory

After private password entry, stale readiness was automatically refreshed
without a source read. At `07:39:09.083Z`, boot identity and artifact matched,
the runner was idle, disk usage was 3.509%, and swap/OOM were zero.

- Inventory operation: `5e678adc-3a0a-42dd-b5e4-1966d68f5f12`.
- Submitted: `2026-09-14T07:39:22.457Z`.
- Report generated: `2026-09-14T07:41:49.990579826Z`.
- Report: 2,947 bytes, SHA-256
  `e217d947f121501c24ae833e593c5c66475a0718461f2d2a5dfc2126b78fafda`.
- GUI report transfer: imported; an independent local hash check agrees.
- Method: `postgresql-repeatable-read-complete-stream`.
- All 18 mappings reached EOF; 1,600,000 vertices / 4,000,000 edges.
- Outcome pass; read-only and source-count checks pass; errors and incomplete
  checks are empty. Counts do not prove unique identity or endpoint correctness.
- Mapped record bytes: 553,598,000; recommended storage range:
  3,484,790,000–9,900,772,000 bytes.
- Observed worker memory: 13,275,136 bytes; guest disk 4%, swap/OOM zero.

Target-review readiness was refreshed again at `07:44:56.640Z`. The 28 source,
placement and assessment regression tests passed, including the assertion
that an on-premises source requires runner ARM calls, not source ARM calls.

## Fresh target plan

The installed GUI saved a secret-reference-only LoadJob and target plan in
the private local OP-PG staging folder and submitted target deployment once.

- Target: `afpg-53625ae3b1554821bfc3`, PostgreSQL 18 / Apache AGE.
- Existing dedicated trial group, Japan East zone 1.
- GP D4ds_v5, 128 GiB, HA disabled for this bounded single-server trial.
- Dedicated subnet `10.246.12.0/24`, private DNS in the existing runner VNet.
  No public database endpoint, peering or source firewall changes.
- Same runner becomes D4s_v5 after separately controlled idle resize;
  application memory remains bounded at 4 GiB.
- Plan SHA-256:
  `3c5514e823041c04bf22aeddb3e3638e7c620231c3e2785ff2f2bc21f6415bba`.
- Reviewed compute: USD 0.736/hour plus accrued/non-compute reserve USD 400.
- Total ceiling USD 800; deadline `2026-09-16T07:14:35.311Z`, unchanged.
- Deployment observed Running at `2026-09-14T07:48:19.852353Z`.

## Safety and remaining gates

Only this runner and the PostgreSQL fixture VM are running; the prior seven
Flexible Servers were confirmed Stopped before target creation. External
Defender/Event Grid/storage actions were observed and not attributed to this
workflow. The new transfer account still has HTTPS/TLS 1.2, anonymous and
shared-key access disabled, and the approved storage-only exception.

The fresh Cost Management query was throttled (429). The latest successful
returned total remains USD 35.32679129750154, subject to billing delay; it is
not the final spend. The conservative reserve and deadline remain binding.

Next: reconcile target completion, apply AGE preload if required, resize the
same idle VM, refresh post-boot readiness, explicitly start a fresh job, then
pass exact counts and the unchanged independent full P1 verifier (all 64
ranges, typed properties, identities, endpoints and canonical root). Preserve
all earlier targets/jobs/evidence. No source or target replay is permitted.
