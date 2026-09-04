# HorizonDB P2 feasibility study progress

## Authorization and source baseline

- Execution authorized: 2026-09-03
- Live infrastructure work started: 2026-09-04
- Branch: `docs/horizondb-age-p2-feasibility`
- Required merge base: tag `v2.1.0`, commit
  `e884019e66d88a18479e558d9ba08ce87d8d712b`
- Production-simulation baseline: commit
  `03965c5334bd59b2cc53ce57c1507bca4d48480a`, the single direct child of
  `v2.1.0` that adds the completed production-simulation qualification
- Working tree: `/Users/rifujita/Git_Managed/agefreighter-horizondb-p2`
- Resource group: `rg-afps-hdbp2-20260904`
- Region: Australia East
- Current state: predeployment gates passed; no VM or database deployed yet

The separate `/Users/rifujita/Git_Managed/agefreighter` worktree remains on
`2.2.0`. Its pre-existing uncommitted Go changes were not modified by this
study.

## Azure predeployment evidence

The following checks were observed against subscription alias
`MCAPS-Hybrid-REQ-51508-2023-rifujita` on 2026-09-03/04:

- `Microsoft.HorizonDB` registration state: `Registered`
- Azure CLI: 2.90.0
- HorizonDB CLI extension: 1.0.0b3
- HorizonDB provider API versions: `2026-05-01-preview` and
  `2026-01-20-preview`
- HorizonDB region availability: Australia East present
- HorizonDB live PostgreSQL capability: majors 17 and 18 present
- HorizonDB live 8-vCore capability: available on Intel and AMD, 8,192 MiB per
  vCore; Intel is frozen for the primary comparison
- VM regional quota: 100 total vCPUs, 100 DSv5-family vCPUs, and 100
  EIBDSv5-family vCPUs; current use was zero
- VM sizes found in Australia East: `Standard_D8ds_v5` and
  `Standard_E8bds_v5`, each with 8 vCPUs
- Flexible Server `Standard_E8ds_v5`: 8 vCores, 8,192 MiB per vCore, zones
  1/2/3, and SameZone/ZoneRedundant HA available
- Dedicated resource group creation: succeeded

The public HorizonDB extension catalog still lists AGE 1.6.0 for PostgreSQL 17
only. The live capability has advanced to PostgreSQL 18, so AGE 1.7.x on that
major remains a hard SQL preflight gate before any timed P2 load.

## Validation evidence

Local validation passed:

- `git diff --check`
- JSON parse for both committed parameter files
- ShellCheck for the changed validation script
- `go test ./production-simulation/...`
- Bicep compilation for `main.bicep`, `horizondb.bicep`, and
  `horizondb-private-endpoint.bicep`

Bicep emitted expected `BCP081` warnings for the preview HorizonDB resource
types because the installed Bicep type bundle has no static schema for them.
ARM server-side validation then returned `Succeeded` for both deployments.

The reviewed `what-if` results contain creates only:

| Deployment | Creates | Modifies | Deletes |
| --- | ---: | ---: | ---: |
| Flexible Server control, sources, loader, network, and Key Vault | 31 | 0 | 0 |
| HorizonDB cluster and AGE parameter group | 2 | 0 | 0 |

The HorizonDB Private Endpoint is intentionally deferred until the cluster
exists and Azure returns its actual Private Link group ID and required DNS
zone. A second `what-if` is mandatory before that endpoint is deployed.

## Next gate

Commit and push the reviewed harness, deploy the two validated templates with
one generated PostgreSQL credential, query the HorizonDB Private Link metadata,
and deploy the private endpoint after its separate `what-if`. Then run the SQL
version/AGE privilege gate before source preparation or any timed load.

