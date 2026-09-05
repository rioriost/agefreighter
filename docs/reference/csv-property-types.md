# Explicit CSV property types

CSV values remain strings unless a mapping explicitly supplies `propertyTypes`.
Keys refer to destination property names in `properties`, not source column names.
Both vertex and edge mappings accept the same optional field:

```yaml
properties:
  source_key: key
  score: score
  active: active
  tags: tags
propertyTypes:
  source_key: int64
  score: float64
  active: boolean
  tags: string[]
```

Supported types are `string`, `int64`, `float64`, `boolean`, and their `[]`
array variants. Arrays use JSON cell content (with ordinary CSV quoting), not
delimiter-separated lists. Array items must have the declared type; null items
and nested arrays are rejected. Booleans are exactly `true` or `false`. Integers
must fit signed 64-bit range; floats must be finite. No type inference occurs.
The configured CSV `nullValue` is applied before conversion, including for arrays.
Conversion failures follow the configured malformed-record/reject policy.

Changing types invalidates the CSV mapping fingerprint, so a saved checkpoint
cannot be resumed with different type semantics. Omitted types preserve the
legacy fingerprint representation and string behavior.

## Strict verification for automation

`agefreighter verify --target job.yaml JOB_ID --counts --require-complete`
returns a nonzero exit status if deep verification is failed **or incomplete**.
Evidence is still rendered or written before that error is returned. The flag
requires `--counts` or `--integrity`; catalog-only validation is not enough.
The default remains backward compatible (incomplete reports alone do not cause
a nonzero exit status). Consumers must also check the report outcome and job
identity. Exact counts do not establish property equality; P1 qualification adds
the complete independent canonical digest comparison.
