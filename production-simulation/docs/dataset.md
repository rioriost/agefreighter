# Supply-chain fixture

## Selection policy

The preferred input is a synthetic reconstruction of an actual migration's
structure. Only schema and aggregate distributions may be used: label/type
counts, degree histograms, property types, null rates, value-size percentiles,
and time distributions. Production values, identifiers, free text, hashes of
production values, and reversible samples are prohibited.

When those aggregate statistics are unavailable, `fixturegen` uses the model
below. It represents suppliers, facilities, products, orders, shipments, lots,
locations, carriers, and customers with correlated temporal activity. It is
designed to exercise migration behavior, not to benchmark supply-chain queries.

## Full-scale cardinality

| Vertex label | Count |
| --- | ---: |
| Supplier | 4,000,000 |
| Facility | 2,000,000 |
| Product | 20,000,000 |
| PurchaseOrder | 45,000,000 |
| Shipment | 35,000,000 |
| Lot | 50,000,000 |
| Location | 1,000,000 |
| Carrier | 100,000 |
| Customer | 2,900,000 |
| **Total** | **160,000,000** |

| Relationship type | From -> to | Count |
| --- | --- | ---: |
| SUPPLIES | Supplier -> Product | 40,000,000 |
| PRODUCED_AT | Product -> Facility | 30,000,000 |
| PLACED_WITH | PurchaseOrder -> Supplier | 45,000,000 |
| CONTAINS | PurchaseOrder -> Product | 100,000,000 |
| FULFILLS | Shipment -> PurchaseOrder | 45,000,000 |
| ORIGINATES_AT | Shipment -> Facility | 35,000,000 |
| DESTINED_FOR | Shipment -> Location | 35,000,000 |
| CARRIED_BY | Shipment -> Carrier | 35,000,000 |
| INCLUDED_IN | Lot -> Shipment | 35,000,000 |
| **Total** | | **400,000,000** |

Smaller phases preserve these proportions using deterministic largest-remainder
allocation, so every phase has its exact declared total.

## Generated properties and topology

Each vertex has a globally unique signed 64-bit `source_key`, a stable
`external_id`, type-specific name, region, timestamp, state, boolean, numeric
score, string and integer arrays, and a variable-width description. Each
relationship has a unique `source_key`, stable `relationship_id`, timestamp,
quantity, state, distance, and variable-width notes. Text widths include common
32-byte values and progressively rarer 256-byte, 2 KiB, and 8 KiB values so the
test does not collapse into an unrealistically narrow integer-only graph.

The fixed seed controls every value and endpoint. Endpoint selection includes
skew for suppliers, products, facilities, and carriers so a small population
becomes high degree. The fallback generator supplies deterministic time,
region, lifecycle, Unicode, null-property, and value-width distributions
without using production data. The P1 calibration decision and its limitations
are recorded in [`fixture-calibration.md`](fixture-calibration.md).

The generator emits headerless, sharded Neo4j bulk-import CSV plus separate
header files. A manifest records phase, seed, cardinalities, byte counts, each
file's SHA-256, and a deterministic root digest. Existing output directories
are never overwritten.

## Identity and ordering contract

- Vertex and relationship `source_key` values are unique and increasing within
  each mapping and fit signed 64-bit integers.
- `external_id` and `relationship_id` are stable across regeneration and both
  Neo4j versions.
- Planner-usable Neo4j indexes are created on `source_key` for every label and
  type: B-tree indexes on Neo4j 4.4 and range indexes on Neo4j 5.26. Neo4j
  4.4 range indexes are migration-preview structures and cannot serve Cypher
  queries.
- All indexes must be ONLINE and the migration query plan must show indexed
  keyset access before the timed run.
- The same manifest root must be imported into both source versions.

## Independent target verification

The agefreighter built-in count and integrity checks are retained. In addition,
P1 through P3 export canonical records by fixed `source_key` ranges. Each range is
sorted, normalized by type, and compared with its generator digest. A Merkle
root identifies the complete graph while allowing a mismatch to be localized
without holding the entire graph in memory.

`cmd/rangedigest` implements the generator-side and target-side streaming
exporters. Its canonical encoding preserves integer/float distinctions, sorts
property names, includes relationship endpoint identities, and produces
100,000-record leaves by default. The implementation is unit-tested and has a
P0 target proof; P1 must still pass it before P2 is authorized.
