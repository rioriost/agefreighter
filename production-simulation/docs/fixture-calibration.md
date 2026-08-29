# P1 fixture calibration decision

## Decision

P1 may use the deterministic fallback supply-chain fixture unchanged. It is
suitable for correctness, streaming-memory, storage-growth, and first-order
capacity measurements. It is not evidence that a particular customer's graph
will complete in the same time, and it is not a query-performance benchmark.

No production values, samples, or approved aggregate distributions from the
historical 160-million-vertex / 400-million-edge migration are available in
this repository or in the retained P0 evidence. Inventing a calibration from
memory would be less defensible than keeping a fixed, reviewable synthetic
model. If privacy-approved aggregates become available, they receive a new
fixture version and a new baseline; an in-place change to an existing phase is
not permitted.

## Represented migration stresses

The fixture deliberately covers the dimensions that materially affect this
migration path:

- nine vertex labels and nine relationship types, with the full-scale totals
  fixed at 160,000,000 and 400,000,000;
- relationship density of 2.5 edges per vertex and type-specific cardinality
  ranging from 30 million to 100 million at full scale;
- deterministic hub skew on suppliers, facilities, products, and carriers;
- signed 64-bit source keys, stable external identities, booleans, integers,
  floating-point values, timestamps, arrays, Unicode, and absent optional
  status properties;
- text-width buckets of 32 bytes (90%), 256 bytes (9%), 2 KiB (0.9%), and
  8 KiB (0.1%), exercising both common and long-property rows;
- five years of event timestamps, six geographic values, and lifecycle/status
  values, without copying a production identifier or value.

P1 preserves the same logical ratios as P3, so it can expose per-label mapping,
endpoint resolution, batching, catalog, and canonical-verification failures
before the expensive phases.

## Known deviations and interpretation limits

- The degree distribution uses a deterministic two-sample minimum for selected
  endpoint labels; it is skewed but is not fitted to an observed power law.
- Property null rates, multi-label vertices, self-loops, duplicate parallel
  edges, deeply nested values, and customer-specific supernodes are not
  calibrated to historical aggregates.
- Values have controlled entropy and compressibility. Storage and network
  coefficients must therefore be measured again at P2 before sizing P3.
- Neo4j store fragmentation, long-running update history, and deleted records
  are not reproduced by offline bulk import.

These deviations do not block the synthetic P1 qualification. They do block
using P1 throughput as a customer commitment. P2/P3 promotion remains governed
by measured storage, WAL, throughput, cost, and the stop criteria in
[`acceptance.md`](acceptance.md).
