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

Stage 1 and the guest execution half of stage 2 are implemented and locally
tested. Extension-side operation persistence/provenance/import and stages 3–4
remain pending. No GUI button or installed remote capability is enabled by these
local source changes alone.

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

## Stage 2 guest boundary and local TLS qualification

The runner now implements a distinct `postgres-catalog` action and its private
CLI command. New builds advertise `postgresql-catalog-v1`; no previously installed
binary gains the capability. The extension does not dispatch it yet.

- A bounded connection-only configuration contains schema version, reviewed
  host/port/database/user, explicit schemas and the reviewed custom-CA SHA-256
  (empty for system trust). There is no placeholder LoadJob, target, arbitrary
  query or caller-supplied file path. Missing/duplicate/unknown JSON fields and
  invalid scopes fail closed.
- The protected URI must match that reviewed identity and use `verify-full`
  plus a 15-second connection timeout. Alternate hosts, extra libpq options,
  unreviewed certificate paths and additional credentials are rejected before
  submission. A custom CA must match its reviewed digest, be certificate-only,
  and be staged only in the operation directory; the child rechecks its digest.
  Ambient `PG*` connection settings are rejected by the private CLI.
- The existing boot check, global dispatch lock, idle/storage/swap/OOM admission,
  create-only operation, durable state-before-start and no-replay worker claim
  apply. Configuration/boot/operation identity and the exact report SHA/length
  remain retained. The reader has its two-minute transaction envelope; the
  catalog child has a three-minute process deadline, inside the existing
  disabled-at-boot, 4 GiB/no-swap systemd worker envelope.
- A catalog-specific report validator requires the exact reviewed schemas,
  complete metadata, every safety flag, unique tables/columns/constraints and
  bounded PK/FK structure. It cannot accept inventory/counts reports instead.
  Chunked/exported report transport uses the existing sealed-byte integrity
  checks. Failed processes or wrong-scope reports never become a successful
  catalog operation. Protected credential/CA files are removed on normal worker
  return, including failures; retained evidence does not authorize replay.

Validation on September 18 (local only):

- Repository-wide `go test ./...` passes; integration suites without their own
  opt-in fixture settings remain skipped. Focused PostgreSQL/CLI/runner/tools
  race tests and vet pass. Linux AMD64 CLI/tools cross-builds pass.
- Both catalog integration tests run against the dedicated local PostgreSQL
  **18.6** container. The CLI test uses actual server TLS with a short-lived
  fixture CA and certificate containing the container IP. It confirms TLS in
  `pg_stat_ssl`, retrieves two real PK/FK tables, validates the complete artifact,
  rejects the server after removing the trusted CA, and preserves source rows.
  The broader metadata/privilege test also passes the new report decoder.
- Runner orchestration tests verify boot and connection rejection, CA binding,
  busy/storage/swap/OOM rejection, duplicate dispatch and worker refusal,
  terminal failure handling, seal retrieval/tamper rejection and secret removal.
  Their subprocess success payload is explicitly a fixture, not Azure or
  systemd execution evidence.
- All **404 extension unit tests**, typecheck and build still pass. No extension
  host or signed-in GUI qualification is claimed for this new operation.
- Existing CSV tests initially failed because this Mac's filesystem was above
  80% use. Capacity is now injected only through an unexported test seam; real
  guest admission continues to read its own filesystem and enforce the same
  80% ceiling. Boundary/invalid/missing-capacity tests verify rejection. No files
  were deleted and no live safety gate was disabled to make tests pass.

The local fixture is stopped again with its disk retained. Its retained TLS
certificate is test-only, expires after one day, and must be regenerated if the
container IP changes. No Azure calls, guest upgrade, operator extension install,
workflow mutation, migration or Marketplace publication occurred. The 70-file
operator-store digest above and installed extension bundle SHA-256
`e3a8fb0aa518b8d8c339aef8aa1d3d861ccfdc82baa7fec32fc2afeef9964ac1`
remain unchanged.

Next: add a separate extension-side catalog intent/controller, capability and
fresh-readiness gates, exact reviewed-configuration/boot/artifact/report binding,
and sealed report import. Then expose explicit schema discovery and selected
recommendation adoption without overwriting manual/unsaved mappings. Counts
inventory must still be rerun after accepted mapping changes. B04 remains
**partial**, not a new installed-GUI or Azure qualification.
