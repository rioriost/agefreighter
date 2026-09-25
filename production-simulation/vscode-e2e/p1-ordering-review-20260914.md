# P1 ingestion-order-independent verification review

The AZ-COSMOS job is committed with passing complete counts; its retained
independent verifier failed with `target-digest/source-key-order`. Cosmos label
queries have no ordering clause. Allocated AGE graph IDs are not fixture keys.
No source query, migration, graph, credentials, or loader installation is changed.

## Correction and limits

- P1 alone buffers one label's canonical records, sorts by the visible typed
  `source_key`, and feeds the unchanged range hash contract. Per-label payload
  is capped at 512 MiB and rows at the expected count (at most 4M). The process
  remains limited to 4 GiB, no swap and 25 minutes. P3 stays streaming.
- Properties retain canonical type checks; edges retain endpoint resolution;
  duplicate keys fail even at a digest-range boundary. All 64 leaves and the
  frozen root must still agree. A counts pass is not sufficient.
- The installed GUI offers an explicit corrected-verifier action only after
  the retained ordering diagnosis, fresh pinned health and budget checks.
  A new operation archives the failed qualification in local history, preserves
  diagnostic evidence, and moves the old inactive marker into its original
  evidence directory. It never retries migration or edits the committed graph.
- The guest checks the exact old marker, boot, loader hash, storage and absence
  of workers under the same diagnostic lock before retaining that marker.
  New verifier artifacts, logs and result use a new create-only directory.

## Review and tests

Reviewed against weakening comparison, unordered duplicates, excessive memory,
P3 regression, accidental migration replay, lost failure evidence, unconfirmed
ARM submission and credential disclosure. Tests cover order-independent roots,
cross-range duplicates/descending keys, row/byte caps, cancellation, unchanged
streaming selection, failed-diagnosis admission and retained-marker handling.
`go test ./production-simulation/...`, extension type checking and all 193 unit
tests passed. Installed-GUI requalification of the same job is still required.
