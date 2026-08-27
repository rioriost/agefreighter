# Cosmos DB integration environment

The azd project provisions only the Azure Cosmos DB for NoSQL resources used by
the live source-connector tests. agefreighter and Apache AGE continue to run
locally.

## Environment

Create or select an azd environment, then set the approved Azure context and
developer access boundary:

```sh
azd env new cosmos-dev
azd env set AZURE_SUBSCRIPTION_ID <subscription-id>
azd env set AZURE_LOCATION japaneast
azd env set AZURE_COSMOS_PRINCIPAL_ID <entra-object-id>
azd env set AZURE_COSMOS_DEVELOPER_IP <public-ipv4>
```

The deployment creates a serverless, single-region account with Session
consistency, local/key authentication disabled, TLS 1.2 minimum, and public
access secured by an enforced Network Security Perimeter. The perimeter allows
inbound access only from the configured IPv4 address. The principal receives
the Cosmos DB Built-in Data Contributor role at account scope.

## Preview and provision

Validation and preview are required before provisioning:

```sh
azd provision --preview
azd provision
```

Export the non-secret deployment outputs for the live test:

```sh
set -a
eval "$(azd env get-values | grep '^AGEFREIGHTER_COSMOS_TEST_')"
set +a
export AGEFREIGHTER_AGE_TEST_DSN='postgres://...'
go test ./internal/app \
  -run '^(TestCosmosLiveIntegration|TestCosmosSourceModeMatrixIntegration)$' \
  -count=1 -timeout=45m -v
```

The test seeds run-scoped documents across several logical partitions and
removes only those exact documents afterward. It exercises a committed-batch
resume, verification, and atomic replacement against local Apache AGE.

## Controlled CI

The `Azure integration` workflow runs weekly or by explicit dispatch on a
self-hosted Linux runner carrying the `agefreighter-azure` label. The runner
must be inside the Network Security Perimeter ingress boundary and provide
Docker. Its protected `cosmos-integration` GitHub environment defines:

- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_SUBSCRIPTION_ID` for an
  OpenID Connect federated identity;
- `AGEFREIGHTER_COSMOS_TEST_ENDPOINT`;
- `AGEFREIGHTER_COSMOS_TEST_DATABASE`;
- `AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER`;
- `AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER`.

The federated identity receives only Cosmos DB Built-in Data Contributor on the
test account. The workflow requests `id-token: write`, uses no client secret,
does not provision resources, and removes only run-scoped fixture documents.

Network Security Perimeter changes can take several minutes to propagate. If
the developer's public address changes, update
`AZURE_COSMOS_DEVELOPER_IP`, preview, and provision again.

## Cleanup boundary

`azd down` deletes the dedicated resource group and is destructive. Do not run
it without explicit approval. If cleanup is approved, first select the exact azd
environment and inspect `AZURE_RESOURCE_GROUP` with `azd env get-values`.
