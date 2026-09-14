# AZ-COSMOS r1 installed-GUI qualification

Status: **draft; paused at the new storage-account access confirmation**.
No new Azure deployment, source inventory or migration has been submitted.
Overall qualification remains 6/9; the earlier headless Cosmos inventory is
not a guided migration pass.

## Scope and live preflight

- Workflow: `7b79f05d-1dc1-40a6-b3dc-6c8129d4e0c1`.
- Installed VS Code GUI selection: Azure Cosmos DB for NoSQL / Azure,
  approved subscription / `rg-af-vscode-p1-20260905-a`, then Discover and
  account `afcosmosp120260907`.
- Runner placement: Japan East / zone 1, initial B2s_v2, retained private
  trial VNet runner subnet. The Cosmos data region is checked separately from
  its account metadata location; no source logical-zone equivalence is claimed.
- Source account: provisioning Succeeded, public network access Disabled,
  local/key authentication disabled. Container `p1/graph` remains at autoscale
  maximum 4,000 RU/s. The retained fixture has 1.6M vertices / 4M edges; fresh
  GUI inventory and full target canonical verification are still required.
- No group locks were returned. Budget remains USD 800 and the renewed
  deadline remains `2026-09-16T07:14:35.311Z`; neither is reset for this route.
  A fresh Cost Management query returned HTTP 429 and was not retried.
  The last returned USD 36.7436452084183 is delayed billing, not final spend;
  retain the conservative USD 400 accrued/non-compute reserve.

The matching published 2.4.0 Linux release is unavailable. The GUI correctly
created only a local draft and submitted no VM deployment. The same reviewed
development loader used by AZ-PGFS will be selected after transfer storage is
prepared; do not substitute an unreviewed build.

## Current GUI handoff

The source form has unsaved basic entries for `az-cosmos-p1-r1`, namespace
`p1`, host `afcosmosp120260907.documents.azure.com`, database `p1`.
All explicit P1 mappings still need to be entered and reviewed. No password or
account key is required; a later, separately reviewed assignment will grant
only Cosmos Built-in Data Reader to the new owned runner identity.

The native confirmation proposes storage account `af7b79f05d1dc140a6b3dc6c`
in the existing trial group, Japan East, Standard LRS. It grants the signed-in
user Storage Blob Data Contributor on **this new account only**. Its HTTPS
endpoint is network-public; anonymous access and shared keys remain disabled.
It does not expose the Cosmos source. The agent paused without selecting
**Create storage and scoped role**, because this adds access on a new scope.

## Remaining sequence

1. Approve and reconcile dedicated storage; preserve secure defaults and use
   only the already authorized trial-storage exception if policy requires it.
2. Upload the unchanged, hash-pinned Linux artifact; approve and deploy the
   private discovery runner; verify guest readiness and budget/health gates.
3. Review all 18 explicit Cosmos P1 mappings and the read-only identity grant.
   Preserve the source-immutability window: Cosmos does not provide one
   transactional snapshot across the complete mapped source.
4. Run complete inventory, import its hash-verified report, review sizing and
   provision a new private AGE target; resize the same runner.
5. Migrate once, require complete counts verification, then compare all 64
   P1 ranges / typed properties / identities / endpoints and canonical root.
6. Only on full GUI qualification, update coverage and stop route compute;
   preserve all data and evidence. No P4, release or unrelated deletion.
