# Independent network resource group target implementation

September 18, 2026. Local implementation follow-up to the
[placement audit](placement-negative-gui-20260918.md). **Not installed or
Azure-qualified.** B01 remains partial; this does not close PostgreSQL key/FK
recommendation work (B04).

## Reviewed design and scope

Keep the runner VM/NIC, private DNS zone and Flexible Server in the selected
migration group. The existing runner VNet may belong to another existing group
in the same subscription. Add only a new delegated subnet there using a scoped,
incremental nested ARM deployment. The parent server depends on that child
deployment's full identity. Do not create groups, redeclare the VNet, infer
peering, expose public access or accept a different network subscription.

This follows the documented ARM
[same-subscription resource-group scope](https://learn.microsoft.com/en-us/azure/azure-resource-manager/templates/deploy-to-resource-group#scope-to-resource-group-in-same-subscription).
Review found that simply removing the old same-group guard would deploy the
subnet into the wrong scope; the change therefore covers template generation,
preflight, review, what-if, reconciliation and preload repair together.

## Safety and compatibility

- Bind the runner input and both deployment identities/templates to the preview
  hash. Display the new subnet's full ID and separate network group/deployment
  in the native approval. Existing same-group plans retain their template shape.
- Both groups must exist and be readable. Existing VM ownership, NIC placement,
  readiness, quotas, pricing, subnet overlap, budget and deadline gates remain.
  No permissions are granted automatically. Provider what-if must succeed at
  both scopes before the parent deployment is sent.
- Check absence of all seven new leaf resources and both deployment identities.
  Preview the network-only child separately without database credentials, then
  require expanded proof of every leaf in the parent preview. An optional exact
  child deployment wrapper is accepted only as `Create`; missing/unexpanded
  subnet evidence, unexpected changes, duplicates and overwrites fail closed.
- Persist one parent intent before one PUT. ARM executes the reviewed child;
  the client never submits it independently. Lost acknowledgement becomes
  `unknown`, and reconciliation uses only reads. Neither deployment is retried.
- Parent success alone cannot mark the cross-group target provisioned: require
  exact parent operation coverage, the exact child deployment's success and its
  single successful subnet operation. Missing/foreign/failed child evidence
  leaves the result unknown. Narrow `ServerIsBusy` preload repair audits both
  scopes before permitting its one existing approved configuration write.
- Old retained same-group workflows need no migration. No accepted workflow,
  graph, disk, source data, credentials or installed extension was changed by
  this development batch.

## Local checks and remaining qualification

Type checking, build and all **398 unit tests** pass, including seven new test
cases covering template scopes, changed preview input, separate what-if gates,
no replay after acknowledgement loss, child-operation coverage, scoped repair,
group access failure and cross-subscription rejection. Test controls use
synthetic ARM responses; they are not live deployment receipts.

All **13 isolated native Extension Host regression tests pass** against the
installed VS Code executable at 09:41 UTC. This existing host suite checks
activation, editors and verification panels, not the new two-scope deployment.
Its profile is separate from the Azure-signed-in operator profile.
An initial launch attempt used the obsolete `MacOS/Electron` filename and did
not start; the actual installed executable is `MacOS/Code`.

The 70 retained operator files still have aggregate filename/content SHA-256
`fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.
The installed operator bundle remains SHA-256
`e3a8fb0aa518b8d8c339aef8aa1d3d861ccfdc82baa7fec32fc2afeef9964ac1`.

Remaining: install an approved pinned candidate, inspect the exact native
two-scope approval, and run a freshly authorized/budgeted isolated Azure trial
through actual what-if, target provisioning, same-VM migration and full P1
canonical verification. Confirm actual nested-operation/what-if response shape
without weakening coverage checks. Test live scope-denied and occupied-subnet
rejections. The earlier bounded cloud session expired; it is not renewed by
this local implementation. B01 cannot be marked PASS from these tests.
