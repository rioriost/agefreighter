# Headless source preparation checkpoint — 2026-09-07

Status: **source-fixture and implementation work in progress; not a guided-path qualification**.

This checkpoint advances work that does not require access to the installed VS
Code GUI or macOS SecretStorage. It uses only the authorized subscription
`MCAPS-Hybrid-REQ-51508-2023-rifujita`, resource group
`rg-af-vscode-p1-20260905-a`, USD 800 ceiling and deadline
`2026-09-09T08:55:00Z`. No resource outside that scope was changed. AZ-N526's
deallocated runner/source, stopped target, private workflow state and target
credential were not changed or substituted.

## Dedicated P1 source fixtures

| Source | Current evidence | Cost state | Qualification boundary |
|---|---|---|---|
| PostgreSQL 18 Flexible Server | Preparation r3 completed with 18 mapped tables, 5,600,000 rows, private networking, verified TLS and a read-only source role | Server stopped after evidence collection | Fixture ready; AZ-PGFS GUI migration and canonical verification not run |
| PostgreSQL 18 on Linux VM | r5 loaded 18 tables / 5,600,000 rows. The reviewed r6 script separates admin and reader credentials, pins the official image digest, checks DNS/IP certificate SANs, and replaces the initialization container so its final config has no password environment or secret mount | Shared preparation VM remains running only while Cosmos work is active | r6 live result pending; AZ-PGVM and OP-PG not qualified |
| Cosmos DB for NoSQL | Private Endpoint, private DNS, managed identity, local-auth disabled, public access disabled, continuous backup, `/partitionKey`, and 5,600,000 deterministic P1 documents. r1 and r2 failures are retained separately. r2 wrote every document; only the unsupported gateway-side cross-partition aggregate failed | Autoscale remains temporarily at the load setting while exact r3 verification runs | Fixture result pending; AZ-COSMOS GUI migration and canonical verification not run |

Cosmos r3 uses idempotent Upsert with the same document IDs. Exact verification
uses the Go SDK's supported simple cross-partition projection, drains every
continuation page and counts rows locally. It does not treat a loader-side write
counter as remote proof. The fixture archive was also repacked without macOS
extended attributes and uploaded under a new immutable object name; old objects
and failed-run evidence remain intact.

## Headless product checks completed

- Field-generated configurations passed the actual Go CLI validator for Azure,
  on-premises and other-cloud Neo4j; Azure, on-premises and other-cloud
  PostgreSQL; Cosmos explicit and Gremlin documents; and local CSV (9/9).
- Extension typecheck, 162 unit tests, compile and VSIX packaging pass. The
  packaged preview is not installed into or reloaded by the user's VS Code while
  the operator is unavailable.
- Neo4j/PostgreSQL custom CA handling is implemented with certificate-only PEM
  validation, local SHA-256 review binding, protected guest transport, strict
  hostname verification and transient guest staging. Claimed workers erase the
  protected transport on normal success and failure paths.
- Cosmos workflows now require a separately previewed and approved, account-
  scoped built-in Data Reader assignment for the owned runner managed identity.
  Intent is persisted before PUT; unknown submission is reconciled by GET and is
  never replayed. No account key or write role is used by the guided source path.
- PostgreSQL complete inventory/migration uses one exported repeatable-read
  snapshot. Cosmos discloses and requires the source-immutability window. Both
  remain subject to full target verification and independent canonical digest.

An isolated Extension Host run was intentionally stopped after it opened an
Electron window and did not finish without desktop interaction. This is not
recorded as a pass or a product failure; the non-GUI unit, contract, compile and
package checks above are the current evidence.

## Remaining work

1. Finish and seal Cosmos r3; then reduce its throughput and replace the fixture
   loader's temporary Data Contributor assignment with Data Reader after a GET
   identity check.
2. Run PostgreSQL-on-VM r6 on the now-idle preparation VM, collect its sanitized
   security/count/TLS evidence, then deallocate the VM.
3. Commit and push the reviewed implementation and redacted checkpoint, build a
   clean commit-pinned Linux guest archive, and run the non-GUI CI suite.
4. When GUI access returns, install/reload the matching VSIX and execute the
   remaining path-specific source assessment, target, migration and full P1
   canonical verification flows. No headless preparation result is promoted to
   a guided GUI qualification.
