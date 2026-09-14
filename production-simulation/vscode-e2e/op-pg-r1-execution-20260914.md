# OP-PG r1 installed-GUI qualification

Status: complete inventory passed and was hash-verified/imported; the fresh
private target and same-VM resize are ready after narrowly scoped AGE recovery.
Migration and canonical verification
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

## Retained deployment failure and scoped repair

The original deployment failed only on `shared_preload_libraries` with
`ServerIsBusy`. The other six resource operations succeeded, including the
server, database and `azure.extensions=AGE`. The server was Ready and private;
preload remained the unchanged system default `pg_cron,pg_stat_statements`.
Concurrent child writes in the original template are a contention risk;
external policy/Advanced Threat Protection operations were also observed, so
exclusive causation is not claimed.

The extension now serializes database, AGE allow-list and preload creation.
For this retained failure only, a separate explicit GUI action validates all
seven original operations, unchanged ownership/placement/capacity/private
network, original plan hash and idle health before one preload-setting PUT.
It cannot replay deployment or recreate resources/reset credentials. Lost
acknowledgement is reconciled with reads only; custom configuration blocks
repair. The original failed deployment is preserved. AGE restart remains a
separate operation. All 188 unit tests and typecheck passed; the revised VSIX
was installed in the existing Mac VS Code.

Installed-GUI repair submitted the single setting update at
`2026-09-14T08:05:25.789Z`; GET-only reconciliation marked it finished.
Azure independently reports `pg_stat_statements,age`, user override, pending
restart. The original deployment still records its failure. Fresh readiness
at `08:04:05.214Z` proved the same idle boot/artifact, 3.511% disk and zero
swap/OOM. The GUI then approved the separate pre-migration target restart.

The restart submitted at `08:06:11.346Z` finished; Azure reports the reviewed
preload value with pending-restart false. The installed GUI separately
deallocated, resized and started the same runner (resize began
`08:07:04.519Z`), then read-only reconciled completion. B2s_v2 became D4s_v5.
Disk/NIC/system identity/security preservation SHA-256 remained
`b641f1a614257b3694ba0f2d5fe2125a80fa1a5e055d1cc33e3a508f6dda21bd`.
Post-boot readiness at `08:10:27.660Z` passed with boot ID
`433e6cc2-c1cf-47ad-a970-abdef7953fde`, unchanged pinned Linux artifact,
idle guest, disk 3.512%, swap/OOM zero. All other eight VMs were deallocated
and all seven earlier Flexible Servers remained Stopped. The current target
is Ready and public access remains Disabled.

The installed-GUI migration preflight passed and its native new-job approval
was accepted. The GUI is now at `Read-only PostgreSQL source password`.
The source password from inventory was not retained; no migration job has
been submitted while awaiting this private entry. Code/evidence fix commit:
`3a8023f`; installed VSIX SHA-256:
`276716ad10114d04c3fa44f1d67ee3d6c3887050b3abd88cc58ec7746b9416ec`.

## Safety and remaining gates

Only this runner and the PostgreSQL fixture VM are running; the prior seven
Flexible Servers were confirmed Stopped before target creation. External
Defender/Event Grid/storage actions were observed and not attributed to this
workflow. The new transfer account still has HTTPS/TLS 1.2, anonymous and
shared-key access disabled, and the approved storage-only exception.

The fresh Cost Management query was throttled (429). The latest successful
returned total remains USD 35.32679129750154, subject to billing delay; it is
not the final spend. The conservative reserve and deadline remain binding.

Next: obtain the private source password for the separately approved fresh job, then
pass exact counts and the unchanged independent full P1 verifier (all 64
ranges, typed properties, identities, endpoints and canonical root). Preserve
all earlier targets/jobs/evidence. No source or target replay is permitted.
