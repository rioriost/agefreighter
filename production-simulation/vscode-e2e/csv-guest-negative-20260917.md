# B08 Linux CSV integrity negative trial

Status: preparation only; no guest result claimed.

## Scope and acceptance

Use a new installed-GUI CSV/local workflow with two separate copies of frozen
P1 Supplier.csv, each 8,797,607 bytes and SHA-256
`0ecaaca3879b11a4bc76835c23f37cda9bfac7d5ed457ad50937170876d861fe`.
Keep all accepted workflows and the September 16 transport-only trial unchanged.
Do not use its repeated-identity large fixture for Linux import or migration.

1. Create only workflow-owned transfer storage and one private B2s_v2 Linux
   runner in the existing trial network, after exact access/software approvals.
   No Flexible Server or source database is needed. Preserve HTTPS-only,
   TLS 1.2, disabled anonymous/shared-key access and no inbound public IP/SSH.
2. Use the already reviewed runner archive `7538981cf0fc6c1bed3a50e6476861e84647a003`,
   37,117,370 bytes, SHA-256
   `6f10538cc70c2125cc669a4d043352efdfce510131e1bec440af3674a4a48d21`.
   Rehash it before selection; no new product binary is needed for this test.
3. Upload both valid copies using normal GUI confirmation. Before faulting,
   preserve original bytes separately and record the exact test Blob's manifest,
   ETag, workflow and file UUID. Intentionally change only the first new test
   Blob's content, keeping its length unchanged. Never change an accepted Blob,
   source fixture, stored GUI manifest or verification result to force a pass.
4. Start the actual GUI import. It must retain a failed operation and partial
   bytes with the wrong digest, publish neither final CSV nor verification seal,
   and remove its transient capability. Reconciliation must not replay it.
5. Prove GUI assessment is unavailable for an unverified mapping. Separately
   import the second unchanged file and verify its full seal as the positive
   control, preserving the failed operation. Only mapped, verified paths may
   pass the receipt gate. No graph load or target deployment is performed.
6. Retain redacted evidence, stop/deallocate the isolated runner and push results.
   A local test or ARM deployment success is not a live negative-path pass.

## Review and gates

The implementation streams the full response through SHA-256 before atomic
no-replace publication and sealing. Same-size corruption specifically tests the
digest, not just Content-Length. An unsuccessful import retains partial bytes;
generic failure diagnostics alone are insufficient, so inspect both the partial
digest and absence of a final file/seal. The successful control must use a
different selected file UUID and must not repair or reuse the failed operation.

Budget remains USD800, with the existing USD600 conservative planning reserve
(not a measured bill). Outer deadline is September 20 16:14 JST. Bound the active
guest trial to at most two hours and deallocate immediately afterward; confirm
the current VM price and quota in the GUI preview. No retired VM or accepted
target should be restarted. Any governance/network exception is exact-account
only and separately authorized, not silently inferred from account creation.

## Preparation evidence

- September 17 01:20 UTC: all eight retained VMs deallocated, all 17 Flexible
  Servers stopped; no active migration to interrupt.
- Installed VS Code 1.138.0 and extension bundle SHA-256 still match the
  qualified cancellation candidate `eb453a1bca36ec7a9f6e1b3dd3cb5c9cc9bf2b78d2cc63c6d74b1dcbdc9243c7`.
- Local `go test ./internal/runner -run 'CSV|Seal' -count=1`: PASS.
- Existing pinned archive was independently rehashed and matches the manifest.
- No Azure resource creation, guest execution or fault injection has occurred.

The ordinary signed-in VS Code profile opened the new workflow through the GUI:
`bd3b6680-1e18-4d78-8f36-f43467a09a0a`, CSV/local, existing trial group,
Japan East/zone 1, B2s_v2 and the existing runner subnet. Native folder selection
saved two file identities without uploading either:

| File | File UUID | Role |
|---|---|---|
| Supplier.csv | `21f56996-d142-4639-89df-64068ec1d742` | isolated negative-case copy |
| Valid-Supplier.csv | `4f5121c6-07d7-4c21-9820-78c9b473ea87` | unchanged positive control |

The native storage approval names `afbd3b66801e184d788f36f4` and an exact-account
Blob Data Contributor grant. A user confirmation is pending for that account,
including its trial-only security tag/network exception; it has not been
submitted. The old trial/account is not reused. No VM preview or deployment is
approved by this preparatory step.

Fresh extension regression: **235/235 tests PASS**, no skipped/cancelled tests.
The pinned build and current tree have no changes in `internal/runner/csv.go`
or its tests, so the local CSV/Seal regression covers the intended guest logic;
it remains local evidence, not a live Linux result. A read-only four-hour RG
activity-log query returned no events at preparation time; this is not proof
that delayed governance events cannot appear later.
