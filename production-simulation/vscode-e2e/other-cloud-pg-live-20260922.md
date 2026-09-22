# B03 PostgreSQL other-cloud selection — offline preparation

September22,2026. Status: **offline preparation and local contracts PASS; no new
workflow persisted, cloud mutation, source read or migration submitted**.
This follows the completed Neo4j other-cloud simulation; it is a separate
PostgreSQL selection and must not reuse the Neo4j credential or qualification.

## Existing fixture and blocking checks

Read-only ARM inspection confirms existing source VM `af-pgvm-source` is
deallocated, Standard_D8s_v5, private IP10.246.1.20 and trial-owned. Existing
Flexible Server source `afpg-p1-source-20260907` is also Stopped; it is not the
selected fixture and must not be started for this trial.

Use the previously accepted OP-PG fixture: endpoint10.246.1.20:5432,
databasep1source, read-only accountagefreighter_reader, verified TLS and frozen
`p1` schema. Laboratory ARM power/health checks remain separate from discovery:
the extension receives only endpoint/database/credentials/CA and mappings.
Prior route workflow53625ae3-b155-4821-bfc3-910cc8cad6df and all its evidence
remain unchanged; a new other-cloud workflow/target is required.

The retained September13 renewed public leaf expired2026-09-20T12:59:55Z.
This is a retained-certificate observation, not a fresh live handshake. The
retained CA is valid until2026-10-06T13:12:43Z, file SHA-256
`0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
Do not weaken TLS or submit inventory against the known expired certificate.
After a new bounded authorization, inspect the exact guest certificate first.
If unchanged, renew only its public leaf with existing CA/key, preserve previous
certificates/evidence, reload the source and verify chain/hostname/IP-SAN/live TLS.
If existing signing material is unavailable or identities differ, stop and report;
do not invent a new CA, reset the reader password or extract private keys.

## Stopped-compute preparation completed

- Opened a genuinely new installed-GUI wizard, preserving accepted workflows.
  Selected PostgreSQL then other-cloud; source Azure subscription/resource
  group/candidate/ARM fields disappear and private-connectivity guidance remains.
  No project-folder request, source access or cloud deployment was triggered.
  The selection is currently unsaved; no workflow identity is claimed.
- Frozen18mapping file SHA-256
  `ac02ab254bb85f929abe033407c4d3358c2addca91c20864c4dce8be4d072e8f`
  matches the reviewed OP-PG properties and endpoints.
- Added an exact-fixture regression: all9vertex/9edge generated mappings preserve
  every property,identity,endpoint and quoted read query; other-cloud/on-premises
  configurations agree and contain only a source DSN environment handle, no ARM
  metadata or embedded endpoint/password.
- Added an actual-Go-validator contract using these18mappings plus the full-P1
  projection admission check. Unit503/503, CLI contracts14/14, typecheck and
  bundle build PASS. These are local tests, not source/migration qualification.
- Runtime production code, installed extension5f93f3c and Linuxd40d6ccc9a4d are
  unchanged. No new development installation is needed for these test-only edits.

## Proposed bounded live sequence — not yet started

1. Obtain a new maximum2hour compute window for this PostgreSQL subcase; the
   completed Neo4j session's14:10UTC deadline is not silently extended or reused.
   Keep cumulativeUSD800, monthlyUSD3750/dailyUSD100 constraints and conservative
   accrued/retention reserve. Refresh billing/governance before cloud mutations.
2. Review a new workflow, dedicated transfer container/storage and private
   B2s_v2 runner in the existing approved trial group/VNet/region. No source
   public access, SSH ingress or peering change. Resolve exact resource names and
   required scoped access/unpublished-artifact approvals before dispatch.
3. Prepare all18mappings and the correct PostgreSQL-on-VM reader credential while
   compute is stopped, with explicit Remember only if the user chooses it.
   PGFS reader credentials are a different source; never infer equivalence.
   Do not start a timed compute session merely to wait for a password.
4. At first compute start set a fixed absolute deadline and scoped monitor.
   Verify source TLS/current fixture and runner disk<80%,memory<=4GiB,swap0/OOM0.
   Run one reviewed full inventory and import its sealed1.6Mvertex/4Medge,
   all18label report. Any failure retains evidence and never automatically retries.
5. Review/save target inputs before deployment; fresh private PostgreSQL18/AGE,
   same-runner resize after assessment, then one explicitly approved GUI migration.
   Preserve prior accepted graphs and jobs. A fresh target/subnet/secret has its
   own exact-scope approval; previous Neo4j target is not repurposed.
6. Import complete counts with zero rejects, then action-time-approved raw-ID
   verifier8a23a5109798 / archiveSHA60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d.
   Require all64ranges and canonical root
   `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
7. Verify exact source/runner deallocated and target Stopped at completion/failure,
   prolonged idle input wait or deadline, preserving resources/disks/evidence.

This is another Azure-hosted endpoint-only simulation, not AWS/GCP certification.
B03 remains partial until this actual PostgreSQL GUI route passes. No new
cloud cost or runtime approval is inferred from local test success.
