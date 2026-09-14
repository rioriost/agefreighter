# Explicit Cosmos property types

Cosmos JSON number spelling alone does not reliably express the intended graph
type: a value intended as floating point may be returned as `1`, not `1.0`.
When the source schema is known, vertex and edge mappings can declare destination
property types. These declarations are optional, not automatic schema inference.

For example, inside a `source.cosmos.vertices` mapping:

```yaml
properties:
  source_key: /source_key
  score: /score
propertyTypes:
  score: float64
```

The corresponding VS Code mapping field is
`source_key=source_key,score=score:float64`. Edge mappings work identically, for
example `distance_km=distance_km:float64`. The guided workflow requires a pinned
Linux runner advertising `cosmos-explicit-property-types-v1` before assessment
or migration with any explicit Cosmos types.

Supported types: `string`, `int64`, `float64`, `boolean`, and their one-dimensional
array forms (`string[]`, `int64[]`, `float64[]`, `boolean[]`). Declarations use
destination property names already present in `properties`. Unknown types and
unmapped names fail validation.

- Undeclared properties keep existing JSON inference and legacy fingerprints.
- Nulls, including null array elements, remain null. Existing missing-field
  handling is unchanged.
- Strings and booleans must have matching JSON types; numeric strings are not
  converted to numbers. Nested arrays are not flattened.
- `int64` requires an exact integer in range, accepting `1.0` or `1e3` but never
  truncating a fraction. `float64` rejects non-finite/overflow/underflow-to-zero
  results and integer-valued inputs that cannot be represented exactly. Ordinary
  fractional decimals use IEEE 754 rounding, not arbitrary decimal precision.
- Numeric conversion bounds its input to 128 characters and exponent magnitude
  400. Invalid values fail conversion; errors do not echo source values.

Types are included in the source fingerprint. Adding or changing a declaration
invalidates old resume tokens. Preserve old jobs, graphs and evidence; create a
fresh job and graph for changed mappings. Do not relabel an already committed
graph or weaken verification to accept a different canonical root.

Declarations express a known source contract. They cannot recover precision or
information already lost before AGEFreighter reads a document. Qualification
still requires full migration and independent post-load verification.
