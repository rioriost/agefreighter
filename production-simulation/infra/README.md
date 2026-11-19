# Azure infrastructure review contract

`main.bicep` describes the zonal, private test data plane. It does not install
Neo4j or agefreighter and does not run a migration. Secure parameters are
deliberately absent from the example parameter file.

The initial values are upper-bound candidates, not authorization to spend. In
particular, two 4 TiB source disks and an 8 TiB Flexible Server are expensive.
For P0 and P1, reviewers should supply smaller phase-appropriate sizes. P2
measurements determine the final P3 sizing.

Compile locally:

```sh
az bicep build --file production-simulation/infra/main.bicep --stdout >/dev/null
```

Create the dedicated, explicitly named resource group after review. Then prepare
a parameter file outside the repository and preview without deploying:

```sh
az deployment group what-if \
  --resource-group <reviewed-resource-group> \
  --name <unique-preview-name> \
  --template-file production-simulation/infra/main.bicep \
  --parameters @<outside-repository-parameters.json> \
  --parameters sshPublicKey="$(< /approved/path/id_ed25519.pub)" \
  --parameters postgresAdministratorPassword='<from-approved-secret-source>' \
  --parameters neo4jPassword='<from-approved-secret-source>'
```

The template enables the `AGE` allowlist entry, but Azure controls the hosted
extension build. Deployment success does not prove AGE 1.7. The SQL version
gate in the runbook remains mandatory before any migration.

Each source version uses a separate database on this one server. After the
server is deployed, `scripts/prepare-target-databases.sh` creates the two
phase-specific databases and installs AGE. Reusing one database for both paths
is prohibited because accumulated identity metadata materially biases the
second measurement.

No deployment wrapper is included before review. This intentionally prevents a
repository command from turning the candidate sizing into live resources by
accident. A reviewed deployment entry point can be added after the first
`what-if` output and budget are approved.

## Rebuilding HorizonDB qualification support

Set `deployTinyTargets=false` on `horizondb-tiny.bicep` to provision only the
runner, network, shutdown schedule, and private Cosmos source. This mode does
not create HorizonDB clusters, private endpoints, or firewall rules. A
replacement NAT allocation does not inherit approval of the previous IP for
HorizonDB ingress. Verify the new static address before requesting that rule.

Deploy `horizondb-scale-safety.bicep` with the new runner identity and a fresh
prefix. Use `protectEvidenceAccount=true` to add an ARM deletion lock to the
new evidence account. Do not reuse the name of a deleted evidence account when
its recovery may still be needed. Publish a fresh guard lease bound to the
actual new disk and verify cost access, private evidence read-back, and
exact-scope cleanup before provisioning paid targets.

For `horizondb-p0-csv-targets.bicep`, supply the approved NAT IPv4 without
the `/32` suffix. The rule uses the same address for both endpoints. Leave
`administratorObjectId` empty on initial provisioning; only enroll the new
runner identity after the writer is Ready and Entra authentication is
persisted as Enabled.
