# agefreighter 2.x Azure Deployment Plan

**Status:** Draft - approval required before Azure artifact generation or deployment

## Scope

- Provision an Azure Cosmos DB for NoSQL test environment for the agefreighter 2.x connector.
- Keep application development and local database integration tests independent of Azure deployment.
- Use Azure Developer CLI (`azd`) as the default deployment workflow unless research identifies a blocker.

## Requirements to Confirm

- Azure subscription and deployment region
- Cosmos DB consistency level, throughput mode, and test-data retention policy
- Authentication model and CI identity
- Budget and cleanup policy

## Proposed Architecture

To be completed after the 2.x architecture report and implementation plan are reviewed.

## Planned Artifacts

- `azure.yaml`
- `infra/` infrastructure-as-code definitions
- Cosmos DB integration-test configuration
- Deployment and cleanup documentation

## Security

- Prefer Microsoft Entra ID and `DefaultAzureCredential`.
- Do not store account keys or connection strings in source control.
- Use least-privilege data-plane roles.

## Workflow

1. Complete and review the 2.x architecture report.
2. Complete and review the implementation plan.
3. Implement and validate the local Go application.
4. Finalize this deployment plan and obtain approval.
5. Generate Azure deployment artifacts.
6. Run Azure validation.
7. Deploy only after explicit approval.

