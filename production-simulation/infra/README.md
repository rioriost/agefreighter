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

No deployment wrapper is included before review. This intentionally prevents a
repository command from turning the candidate sizing into live resources by
accident. A reviewed deployment entry point can be added after the first
`what-if` output and budget are approved.
