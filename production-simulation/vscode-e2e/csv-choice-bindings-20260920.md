# CSV choice binding audit — September 20, 2026

## Scope and result

The defined P1 CSV file/type/null/endpoint choices are now bound to retained
evidence. This audit neither reran migration nor broadened the accepted P1
dataset. Installed extension remains `3dff349`; only regression tests and
documents changed. No Azure start, data upload, source edit or credential access.

Reconnected through the installed VS Code GUI to accepted CSV recovery workflow
`54da6ddd-27d2-45e0-bb68-cf5f352801db` and opened source assessment read-only.
Observed all 18 graph labels paired with the corresponding selected CSV filename,
vertex `external_id`, edge `relationship_id`, all endpoint labels/fields and
explicit scalar/array property declarations. The null marker is `\N`.
Opened the Supplier file dropdown: placeholder plus all 18 selected filenames,
Supplier selected. Cancelled without choosing a replacement or saving settings.

Independent retained-record audit:

- All 18 selected file IDs match their form mappings, exact generated Linux
  upload paths and `verified` transfer seals. Filename equals label plus `.csv`.
- Rebuilding the configuration from the saved fields and selected file IDs is
  structurally identical to the persisted configuration.
- `assertP1Projection` passes for all intended properties and endpoint mappings.
- Existing full canonical P1 qualification remains pass; this audit adds choice
  provenance, not a new full-digest run.
- All 74 operator JSON files remain byte-identical. Aggregate sorted
  filename/content SHA-256 before and after:
  `bbf2f43d743b436d949fdd3fff2560b20cf904d812cae32522d87c5073180e1c`.

## Added regression evidence

Three tests execute the actual source webview script with its DOM harness and
pass emitted fields to the production configuration builder. They cover file-ID
binding despite duplicate display names or reordered options, all eight scalar/
array types, vertex/edge identity and endpoints, three null markers (`\N`, empty,
`NULL`), invalidating review on file/null edits, missing file selections and no
implicit upload/source assessment. These are simulated DOM interaction tests,
not extra installed-GUI trials with duplicate files.

Three additional actual-Go validator contracts cover all eight type declarations
on vertices and edges with the three null markers. This validates generated
configuration, not actual parsing/migration of every possible CSV value.
An initial test expectation compared the parser's null-prototype type map with
a normal object; the assertion now compares its own entries. No production code
change was needed. All **445 unit/controller tests**, typecheck and **13 actual-Go
source contracts** pass.

## Remaining B04 boundary

The accepted P1 CSV choices are evidenced; arbitrary CSV combinations are not
claimed end-to-end tested. PostgreSQL live FK recommendation/adoption remains
open because the accepted P1 schema has no foreign keys. Do not add constraints
to that schema or reinterpret manual edges as discovered FK evidence. See the
[isolated FK proposal](postgres-fk-live-plan-20260920.md) before any live change.
