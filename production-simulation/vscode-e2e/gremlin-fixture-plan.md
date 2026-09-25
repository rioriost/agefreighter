# B05 Gremlin fixture and independent oracle

Scope: local preparation before any new Azure mutation. Preserve the frozen
P1 input, raw-ID canonical root, all existing exports and accepted graphs.

## Representation

- Export all nine vertex labels and nine edge types into a **new** directory.
- Vertex partition: `<label>-<zero-based label ordinal modulo 64>`.
- Edge partition equals its source vertex partition; `_sinkPartition` equals
  the destination vertex partition. Element IDs remain the frozen external IDs.
- Vertex user properties use one `_value` wrapper containing the original
  value (including lists and null). Edge user properties remain flat.
- Exclude transport-only fields; retain all original projected properties.
  Use explicit `float64` for `score` and `distance_km` after JSON transport.
- Retain per-file SHA-256 and row counts and a named representation version.

## Oracle and review

The expected oracle reads the original verified CSV shards directly. It uses
source keys and label first-key ranges to derive partitions, independently of
the exporter's external-ID ordinal parser. Only canonical identity slots are
changed to `[partition, id]`; original properties are unchanged. A distinct
canonical version/root prevents confusion with the old P1 digest. Never strip
partitions from observed records to make the old oracle pass.

An offline reader feeds exported JSON through the production Cosmos iterator,
with bounded, reversed pages and JSON reserialization. Its result is explicitly
`cosmos-gremlin-offline`, not an Apache AGE target result. Check file hashes,
table metadata, counts, complete range coverage, composite IDs and both
endpoints. Fail on property/type/partition/endpoint changes and duplicate or
missing records. Do not contact Azure from this checker.

Review priorities: separate expected/actual derivation; no unbounded graph in
memory; create-only output; no silent projection or precision loss; retain
partial output on failure; validate full P1 only after tiny negative tests.
Generating ordinary CSV alongside Gremlin documents reuses the established
portable exporter and allows an independent unchanged raw-ID CSV check.

## Acceptance

1. Tiny fixture round trip and negative mutations pass locally.
2. Full frozen P1 (1.6M vertices + 4M edges / 64 ranges) agrees in every range;
   record input root, new canonical root, output hashes, duration and memory.
3. Only then extend the read-only target verifier to include retained composite
   identities and actual endpoints, pin supporting Linux/extension artifacts,
   refresh cost/time/ownership/health and perform the separate GUI/Azure trial.

This plan does not mark B05 or B10 complete or authorize a new budget window.
