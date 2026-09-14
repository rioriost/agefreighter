# AZ-COSMOS r2: typed mapping installed-GUI qualification

Updated: 2026-09-14T22:17:03Z (2026-09-15 JST). Outcome: **pending**, not PASS.
Overall GUI coverage remains **6/9**. Old r1 graph/jobs/evidence are unchanged.

## Completed

- Mac unlocked; VS Code window reloaded. Updated installed GUI visibly exposes
  Cosmos `name=field:type` instructions and typed mapping fields.
- Fresh GUI workflow: `d138f4e4-bcf3-40fe-a876-ee9ce062e08a`. Source selection
  used subscription, resource group, Discover, and `afcosmosp120260907`.
  Placement: Japan East / zone 1, existing private runner subnet, initial B2s_v2.
  Cosmos data region, not a logical source availability zone, is checked.
- Source name `az-cosmos-p1-r2`, namespace/database `p1`: all nine vertex and
  nine edge mappings entered and reviewed in the GUI. The persisted form is
  independently exactly equal to `fixtures/cosmos-p1-typed-mappings.json`.
  All `score` and `distance_km` properties explicitly declare `float64`.
- No Azure write, source read, deployment or migration submitted this turn.
  Workflow remains `draft`; no credential entered or extracted.

## Live read-only gates

- All 12 existing VMs deallocated; all 10 Flexible Servers Stopped.
- Cosmos public network Disabled, local/key authentication disabled, private
  endpoint Approved, actual data region Japan East; no RG locks returned.
- External activity includes storage, Defender and Event Grid writes around
  19:33–19:44Z by another principal, plus policy audit results. No controls were
  reverted. The old r1 transfer account still has authenticated public HTTPS,
  no anonymous access, shared keys disabled and TLS1.2.
- Cost Management returned 429 once; not repeatedly retried. Last available
  cost is delayed, not final. USD 800 ceiling, USD 400 reserve and
  `2026-09-16T07:14:35.311Z` deadline unchanged; about 32h57m remain.
  Reviewed combined runner/target compute USD 0.736/hour would add about
  USD 24.25 through the deadline. This is an estimate, not current billing;
  retained Cosmos/storage charges remain separate.

## Storage approval and current connectivity gate

At the user's explicit action-time approval, **Create storage and scoped role**
was pressed at approximately `2026-09-14T22:48:47Z`. Deployment succeeded;
the GUI reconciled storage to ready. Independent ARM reads confirmed the
account-scoped role assignment `db222a8c-5b6d-4cb6-b4af-8740911f87e6` grants
the signed-in user Storage Blob Data Contributor only on this new account.

However, live Public Network Access is **Disabled**, despite the reviewed
template requesting Enabled. The account activity log includes successful
`Microsoft.Authorization/policies/modify/action` at
`2026-09-14T22:48:58.7262832Z`; this supports policy modification during creation.
The GUI explicitly reports `ready — public network: Disabled (provisioning is
not transfer readiness)`. Shared keys and anonymous access remain disabled,
TLS1.2 remains configured. No upload, VM deployment, assessment or migration
has begun. Do not equate provisioning success with transfer readiness.

The current handoff is whether to apply the same trial-storage-only official
`SecurityControl=Ignore` exception plus unchanged expiry and authenticated
public HTTPS used for r1, now to `afd138f4e4bcf340fea876ee` only. No exception,
network re-enablement or policy change has been applied to this new account.
Source networks and authentication must remain unchanged.

### Earlier approval handoff (resolved)

GUI shows **Create dedicated transfer storage and grant your Azure user data
access?** for `afd138f4e4bcf340fea876ee` in `rg-af-vscode-p1-20260905-a`.
It would grant the signed-in user **Storage Blob Data Contributor on this NEW
account only**. Standard LRS/request/egress charges apply; HTTPS is network-public,
anonymous access and shared keys disabled. No source server is exposed.
The final **Create storage and scoped role** button has **not** been pressed.
New security-sensitive access requires action-time confirmation.

## Next

After confirmation, prepare/reconcile storage, select the reviewed Linux
manifest `work/vscode-runner-build.GXKmVv/manifest.json`, upload and pin through
the extension, then review a fresh runner deployment. Implementation commit:
`8a23a5109798ec906109532e4cc6c32308b3c824`; runner archive SHA-256:
`52e1d147a13b86a729f5a993e9e72848dd87a89d0ae50a61f26459f5632444f3`.
Retain old Run Command receipts and never replay the old job. A new runner's
Cosmos Data Reader grant is a separate access decision. Continue GUI inventory,
target/resize, fresh migration, complete counts and independent full 64-range
canonical comparison. Expected root remains
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
