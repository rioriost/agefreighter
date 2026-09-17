# Retained placement, mappings and resize audit

September 17, 2026, approximately 05:34–05:41 UTC. Scope: B01/B02/B04/B07
evidence mapping and one B09 native cancellation. No compute was started, no
cloud resource was modified, and no accepted migration was rerun.

## Placement coverage: confirmed and missing

Read-only inspection of all 18 saved workflow records found Japan East, zone 1,
runner RG `rg-af-vscode-p1-20260905-a` and a subnet in that same RG. Each Azure
source resource ID is also in that RG. The nine accepted base routes therefore
establish the same-RG/same-region/same-zone path, **not** separate migration or
network RGs, zone overrides, unknown source zone or another region. B01/B02 are
partial rather than wholly untested; the unrepresented choices remain open.

The retained completed resize records use B2s_v2 → D4s_v5. They retain the
original input size and separate migration size, rather than pretending the
discovery VM was initially D4s_v5. The historical AZ-N44 result already binds
its exact resize preservation hash to its completed GUI migration:
[AZ-N44 evidence](evidence/az-n44-qualified-20260906.json).

## Live read-only VM identity check

Azure VM GETs at 05:39:21–05:39:23 UTC independently recomputed the same ordered
hash input as `runnerResize.ts`: lowercased persistent OS disk/NIC ARM IDs,
managed identity principal/tenant, disk-controller type, normalized Standard
security profile, region and zone list. Each matched its retained pre-resize
seal, ownership tags and requested D4s_v5 SKU. All three were **deallocated**.

| Workflow | Retained and current SHA-256 |
|---|---|
| `54da6ddd-27d2-45e0-bb68-cf5f352801db` (CSV recovery) | `959b706bd7812e8c70dda1f09600339d1dcc60f57e4eaf04222e1709249a91c9` |
| `8a9ae99e-c621-4a94-afd1-a30ff210a201` (retained first network attempt) | `bbb092b27493316992059c86a7fc00651f6e8bf894447a8aed38d61cc1ac99ac` |
| `b2c7214e-83f5-4613-b378-98d36e0cd97d` (network recovery r2) | `57fdc161fa0ed0bc2d0f626f14b6af7509b24efda0516b3123ecd38d82211dba` |

This checks identity preservation, not fresh running-guest health or an
active-job denial. Deleted runner VMs were not recreated for this audit.

## Installed signed-in GUI observations

Normal installed bundle SHA-256 remained
`3061aea43c315113f47df3ad1da31dc68ad73c27b47a18571ddfb66bf742e929`.
Used the actual command palette and native picker to select CSV recovery
workflow `54da6ddd-27d2-45e0-bb68-cf5f352801db`:

1. `Reconcile resize (read only)` displayed **Reconcile target and guest
   operations before resizing the idle runner.** The saved migration is
   finished; production `gate()` rejects any retained migration before ARM
   access. This qualifies completed-migration refusal, not a running-job test.
2. Selected `Approve next same-VM resize step`, inspected the actual native
   dialog, and clicked **Cancel**, never Approve. The dialog showed the exact
   retained VM, finished resize, D4s_v5 and historical September 16 deadline.
   That old deadline was not extended or treated as current execution authority.
   Cancellation returns before price lookup, persist or resize calls in the
   reviewed handler. No resize/restart was attempted.

All 64 prior workflow/report JSON artifacts remain byte-identical; aggregate
SHA-256 at 05:41:13 UTC:
`917a29e7457ef2b6f235970141cdc4c262c10543329aa04d189e7c18ee4fc8f8`.
The no-write conclusion combines actual UI actions, unchanged records, current
VM identity/power and reviewed early-return code, not live packet capture.

## Mapping evidence bindings

These saved source configurations are linked to prior complete counts and
canonical PASS, not new migrations or newly exercised recommendation choices:

| Route / workflow | Retained mapping |
|---|---|
| CSV-MAC / `83c6b829-acdc-4405-aa2d-fb2f2d99af9f` | Nine vertices × 11 properties and nine edges × 7 properties; explicit string, int64, float64, boolean, string[] and int64[] declarations |
| AZ-PGVM corrected / `c275d043-de93-4b0a-b2b0-59cddd13c84f` | Nine vertices × 11 properties and nine edges × 7 properties; source keys and identity fields explicitly projected; native PostgreSQL types, not CSV type declarations |
| AZ-COSMOS corrected / `d138f4e4-bcf3-40fe-a876-ee9ce062e08a` | Nine vertices × 11 properties and nine edges × 7 properties; explicit float64 declarations; explicit-document mode, not Gremlin |

Saved record SHA-256, respectively:

- `2f7c5a26daff57c1b71c698c3eb9bb505847dcdb299a769b5b584e622db5b52b`
- `be7fde11617334f36b693beb3ff232234450256af1302430681e4f3e11c688ae`
- `02be10fab2a0e1abac267917f8d6f6a3cb51087b4c3196c847ca4ff1119cd756`

## Added regressions and limits

Eighteen new production-resize unit cases passed: submitted/running/finished/
failed migration; submitted/unknown guest command; changed disk/NIC/principal/
tenant/controller/security; ephemeral disk, data disk and multiple NIC layouts;
missing SKU/quota; and both ready phases without renewed approval. They assert
no further write/persist (or zero ARM reads for retained-job gates). The existing
unknown-acknowledgement test still proves GET-only reconciliation.

TypeScript typecheck PASS; full unit suite **273/273 PASS**, no skips. These
inert-adapter regressions do not replace actual active-job/incompatible-live-VM
GUI denial evidence. No production or installed extension code changed.
B07 and B09 remain partial. B04's recommendation adoption/edit choices still
need exact GUI bindings; B01/B02's missing placement choices remain open.
