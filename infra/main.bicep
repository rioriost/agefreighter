targetScope = 'subscription'

@minLength(3)
@maxLength(24)
@description('Azure Developer CLI environment name.')
param environmentName string

@description('Azure region for the integration environment.')
param location string

@description('Microsoft Entra object ID granted Cosmos DB data-plane access.')
param developerPrincipalId string

@description('Public IPv4 address allowed through the Cosmos DB firewall.')
param developerIp string

@description('Static public IPv4 address of the controlled GitHub Actions runner.')
param runnerIp string

var resourceGroupName = 'rg-agefreighter-${environmentName}'
var resourceTags = {
  'azd-env-name': environmentName
  purpose: 'agefreighter-cosmos-integration'
  environment: 'development'
}

resource resourceGroup 'Microsoft.Resources/resourceGroups@2025-04-01' = {
  name: resourceGroupName
  location: location
  tags: resourceTags
}

module cosmosBase 'modules/cosmos.bicep' = {
  name: 'cosmos-base'
  scope: resourceGroup
  params: {
    accountName: 'af${uniqueString(subscription().subscriptionId, environmentName)}'
    location: location
    publicNetworkAccess: 'Disabled'
    tags: resourceTags
  }
}

module cosmosData 'modules/cosmos-data.bicep' = {
  name: 'cosmos-data'
  scope: resourceGroup
  params: {
    accountName: cosmosBase.outputs.accountName
    databaseName: 'agefreighter'
    developerPrincipalId: developerPrincipalId
  }
}

module networkPerimeter 'modules/network-perimeter.bicep' = {
  name: 'network-perimeter'
  scope: resourceGroup
  params: {
    accountId: cosmosBase.outputs.accountId
    developerIp: developerIp
    runnerIp: runnerIp
    location: location
    perimeterName: 'nsp-agefreighter-${environmentName}'
    tags: resourceTags
  }
}

// Apply NSP-controlled access only after the association exists so inherited
// policy can distinguish this account from an unrestricted public endpoint.
module cosmosPerimeter 'modules/cosmos.bicep' = {
  name: 'cosmos-perimeter'
  scope: resourceGroup
  dependsOn: [
    cosmosData
    networkPerimeter
  ]
  params: {
    accountName: cosmosBase.outputs.accountName
    location: location
    publicNetworkAccess: 'SecuredByPerimeter'
    tags: resourceTags
  }
}

output AZURE_RESOURCE_GROUP string = resourceGroup.name
output AZURE_COSMOS_ACCOUNT_NAME string = cosmosPerimeter.outputs.accountName
output AGEFREIGHTER_COSMOS_TEST_ENDPOINT string = cosmosPerimeter.outputs.endpoint
output AGEFREIGHTER_COSMOS_TEST_DATABASE string = cosmosData.outputs.databaseName
output AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER string = cosmosData.outputs.vertexContainerName
output AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER string = cosmosData.outputs.edgeContainerName
