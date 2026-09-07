# Headless source preparation checkpoint — 2026-09-07

Status: **source fixtures and four complete headless inventories passed; not a
guided migration qualification**.

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
| PostgreSQL 18 Flexible Server | Preparation r3 and AGEFreighter complete inventory passed: 18 mapped tables, 5,600,000 rows, private DNS, verified TLS and a read-only source role | Server stopped after evidence collection | AZ-PGFS source path is ready; target deployment, migration and canonical verification not run |
| PostgreSQL 18 on Linux VM | Preparation r7 and both DNS and IP-only AGEFreighter inventories passed: 18 mappings / 5,600,000 rows, fixed image digest, verified TLS, read-only role and no password environment or secret mount | VM deallocated after evidence collection | AZ-PGVM and OP-PG source paths are ready; target deployment, migration and canonical verification not run |
| Cosmos DB for NoSQL | Preparation r3 and AGEFreighter complete inventory passed: 5,600,000 deterministic P1 documents through a Private Endpoint using only the runner identity and account-scoped Data Reader | Autoscale returned to 4,000 RU/s after the bounded scan | AZ-COSMOS source path is ready; target deployment, migration and canonical verification not run |

Cosmos r3 used idempotent Upsert with the same document IDs. Exact verification
uses the Go SDK's supported simple cross-partition projection, drains every
continuation page and counts rows locally. It does not treat a loader-side write
counter as remote proof. It completed at `2026-09-07T01:08:08Z`: all 5,600,000
remote documents and all 18 file counts matched, disk use was 9%, and swap/OOM
were zero. Source and load evidence SHA-256 values are
`36d575024984ed6c3efd1c78ffb3634aaf433de4dbff1e8142e71c6a90b31178`
and `8fa3c5b16daae9fb161ab7fd1f286cb717b7fc4ba8a5f215392cad9195364c8d`.
The temporary Data Contributor was replaced by built-in Data Reader assignment
`401792f5-9386-4524-81f8-7ddb0d03dee5` only after a matching GET; the write
assignment was then deleted and the remaining assignment was checked.

PostgreSQL VM r6 exposed a delayed persistence defect that an immediate
readiness check missed. Five minutes after the final container start, its first
timed checkpoint lost access to PostgreSQL 18's root-created intermediate data
directory. The failed container, logs and evidence were sealed with checksums.
r7 normalizes the complete dedicated PGDATA tree to UID 999, forces a checkpoint
after final startup and does not apply a broad permission change to PGDATA. The
forced checkpoint and a later timed checkpoint at `2026-09-07T01:44:56Z` both
passed. The final container uses
`postgres:18.1@sha256:1090bc3a8ccfb0b55f78a494d76f8d603434f7e4553543d6e807bc7bd6bbd17f`
and has no password environment, secret mount or initialization password file.

## Complete headless AGEFreighter inventories

All scans used the actual Linux amd64 AGEFreighter artifact from commit
`06c0e9f225f2c81db527f519ebba1a42d3d9c7fd`, decoded all mapped properties,
reached EOF with no malformed records and compared all 18 per-label counts.

| Path | Access boundary | Started-finished (UTC) | Result/report SHA-256 |
|---|---|---|---|
| AZ-PGVM source | Private DNS, custom CA, `verify-full`, read-only role | 01:37:38-01:39:21 | PASS; `2d74dbffd8e2aa83535ed35fe5567edf24cdbf9764743dbcbd2e0dac8ebd58e9` |
| OP-PG source simulation | IP and port only, custom CA with IP SAN, `verify-full`, read-only role | 01:56:33-01:58:20 | PASS; `d283d36f6187fc105bd2da4e47addf5c2a3a19bc97f2982d6d219f7dbaf85f82` |
| AZ-PGFS source | Flexible Server Private DNS, system CA, `verify-full`, read-only role | 02:04:27-02:06:16 | PASS; `6bee7b9932b36605be9634b5349d280d511eebf603067a4795d0cc1ebe558b53` |
| AZ-COSMOS source | Private Endpoint, managed identity, Data Reader only | 01:43:50-01:48:39 | PASS; `24d6f3f2038c167386d12a05bf0aad5f96cca07ce5c45b9c4e72db20e2010818` |

Each report returned 1,600,000 vertices, 4,000,000 edges, 5,600,000 total
records and the expected counts for every label. These are source-read and
capacity-evidence qualifications. They are not evidence that any target was
deployed, loaded or canonically verified for those four paths.

## Headless product checks completed

- Field-generated configurations passed the actual Go CLI validator for Azure,
  on-premises and other-cloud Neo4j; Azure, on-premises and other-cloud
  PostgreSQL; Cosmos explicit and Gremlin documents; and local CSV (9/9).
- Extension typecheck, 162 unit tests, compile and VSIX packaging pass. GitHub
  Actions run `34071519322` passed Linux, Windows, macOS, Extension Host,
  packaging and the nine real-CLI source contracts. The packaged 2.4.0 VSIX was
  installed non-interactively into VS Code 1.136.1; no window reload or GUI
  qualification was attempted while the operator was unavailable.
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

An isolated local Extension Host run was intentionally stopped after it opened an
Electron window and did not finish without desktop interaction. This is not
recorded as a pass or a product failure. The independent GitHub Actions
Extension Host job passed.

The commit-pinned Linux archive was uploaded under an immutable blob name with
SHA-256 `b68c726ddadbb5b902e390a7d4c20cf9c731e128f9ef1641ed892a28186c4355`
and size 37,040,215 bytes. The source tree now disables macOS extended
attributes for subsequent archives; the already-qualified archive emitted only
harmless extraction warnings for two provenance attributes.

After committing the retained corrections, a new archive was built from
`e513db56820c52e3d91296e00217e12752f2658d`. Its immutable blob contains only
`agefreighter` and `agefreighter-tools`, has SHA-256
`07522145ab0a92093194cec02424b95934b316f484a74efe092681dafce3edcd`, and is
37,040,133 bytes. A fresh download reproduced that digest and emitted no
extended-attribute warning. This post-run artifact has not been substituted for
the older artifact named in the four completed inventory reports.

The Cost Management snapshot for this resource group was USD
`7.37302738132321` at `2026-09-07T01:00Z`; billing can lag, so the USD 800 gate
remains authoritative. No lock existed. External policy actions against the
test VNet/NSGs were recorded through `00:24Z`; exact networking and source
security state were re-read before mutations.

## Remaining work

1. Prepare separate IP-SAN Neo4j 4.4/5.26 fixtures or an independently reviewed
   TLS server-name design before attempting OP-N44/OP-N526. The retained
   AZ-N526 source and its SecretStorage-dependent workflow must not be modified.
2. When GUI access returns, reload the installed VSIX and execute the
   remaining path-specific source assessment, target, migration and full P1
   canonical verification flows. No headless preparation result is promoted to
   a guided GUI qualification.
