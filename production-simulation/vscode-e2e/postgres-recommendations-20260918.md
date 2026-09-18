# PostgreSQL schema recommendation implementation

September 18, 2026. B04 remains partial. Accepted manual PostgreSQL P1 routes
are preserved and do not qualify automatic schema recommendations.

## Plan and design review

1. Build a bounded, metadata-only PostgreSQL catalog reader and deterministic
   recommendation engine with negative tests. Explicit schemas only; no customer
   row reads, estimates, source writes or inferred permissions. Use one read-only
   repeatable-read transaction and fail closed on timeout/size/coverage bounds.
2. Connect a distinct Linux runner operation and capability, retaining the exact
   request, boot, artifact, operation and report hashes. Do not fake a LoadJob
   with placeholder tables just to pass existing assessment validation.
3. Expose separate schema discovery, report import and explicit recommendation
   adoption in the GUI. Keep existing manual mappings; no automatic overwrite.
   Edits invalidate source review and require fresh inventory before target
   sizing. Catalog evidence is neither row counts nor migration verification.
4. Qualify the pinned implementation locally, then with an approved Linux/GUI
   trial. No old guest advertises the new capability until its implementation
   and installation are actually complete.

Stage 1 is implemented and locally tested. Stages 2–4 remain pending;
no new GUI button or remote operation is enabled by a catalog library alone.

Review decisions: only single-column, non-null, supported-type primary keys
are automatic vertex candidates. Foreign keys produce optional directed
referencing-table-to-referenced-table edge candidates, not inferred business
semantics or automatic bridge-table classification. Reference columns must
match the target's selected primary key. Composite, nullable, unvalidated,
unenforced, temporal or deferrable constraints, RLS/inheritance, missing targets,
ambiguous names and unsupported identities require manual review. Nonstandard
key equality/index classes and nondeterministic string collations also require
manual review: SQL may consider different byte strings equal while graph
endpoint identities do not. This follows the documented
[collation distinction](https://www.postgresql.org/docs/18/collation.html#COLLATION-NONDETERMINISTIC).
Properties
are explicitly selected; discovery never silently copies every column.

Catalog interpretation follows PostgreSQL's official
[constraint catalog](https://www.postgresql.org/docs/17/catalog-pg-constraint.html),
[attribute catalog](https://www.postgresql.org/docs/17/catalog-pg-attribute.html)
and [table catalog](https://www.postgresql.org/docs/18/catalog-pg-class.html).
Use only metadata visible to the authenticated principal. The result must not
claim that unreadable objects, rows or cross-schema references were covered.

## Stage 1 implementation and evidence

`internal/source/postgres/catalog.go` exposes a library operation, not a new
CLI command or remote action. It reads only catalog metadata for 1–16 explicit
non-system schemas, with limits of 64 tables, 128 columns and 64 PK/FK
constraints per table, a two-minute overall deadline, 20-second statement
timeout, two-second lock timeout and 4 MiB serialized report ceiling.
The transaction is repeatable-read/read-only with catalog-only search path;
oversized, missing-schema and failed responses cannot return a complete report.
No source data values, credentials, generated SQL, estimates or target handles
are included. TLS remains the responsibility of the existing protected caller;
the future remote boundary must enforce its usual certificate checks before
using this library.

`core/postgresRecommendations.ts` validates that contract and returns stable,
schema-qualified, bounded labels. Only explicitly selected candidate IDs can
be adopted. Adoption deep-copies inputs, refuses duplicate labels/unknown IDs,
requires matching endpoint vertex mappings and enforces the 64-mapping limit.
It does not replace manual settings. Proposals project the identity property
only, explicitly requiring review of other properties and relationship direction.
The caller still needs to verify operation/source/boot/artifact/report identity;
computing a report hash in this pure function is not an authentication check.

Validation on September 18:

- All **404 extension unit tests**, type checking and build pass. Six new cases
  exercise generated LoadJob compatibility, deterministic labels, conservative
  key/FK rejection, malformed/incomplete reports and non-destructive adoption.
- `go test -race ./internal/source/postgres -count=1` passes with the new catalog
  integration test enabled. Other integration suites requiring their own DSNs
  were not enabled; this is not a full PostgreSQL version matrix.
- A dedicated local Apple Container, `af-catalog-local-20260918-a1`, runs the
  cached official PostgreSQL **18.6**, ARM64, 2 CPUs / 1 GiB, no published host
  port. It is distinct from other running project containers. A restricted
  fixture role confirmed access flags, composite keys, nullable foreign keys,
  `NOT VALID`, RLS, inheritance, nondeterministic ICU text-key collation and view
  metadata. Catalog collection did not
  execute a deliberately failing view or include a marked customer-row value.
  Fixture source row counts remained unchanged.
- The local fixture uses non-production credentials and non-TLS on its private
  container network solely for this integration test. This does not qualify
  remote TLS, custom-CA, runner dispatch or GUI adoption.

No Azure calls, guest installation, operator workflow changes, extension
installation or new capability advertisement occurred. B04 remains partial.
The dedicated local fixture container is stopped with its disk retained.
All 70 operator files remain at aggregate filename/content SHA-256
`fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.
