@description('Existing Cosmos DB account name.')
param accountName string

@description('Azure Cosmos DB for NoSQL database name.')
param databaseName string

@description('Microsoft Entra object ID granted data-plane access.')
param developerPrincipalId string

var vertexContainerName = 'vertices'
var edgeContainerName = 'edges'
var dataContributorRoleDefinitionId = '${account.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'

resource account 'Microsoft.DocumentDB/databaseAccounts@2025-04-15' existing = {
  name: accountName
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2025-04-15' = {
  parent: account
  name: databaseName
  properties: {
    resource: {
      id: databaseName
    }
  }
}

resource vertexContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-04-15' = {
  parent: database
  name: vertexContainerName
  properties: {
    resource: {
      id: vertexContainerName
      partitionKey: {
        kind: 'Hash'
        paths: [
          '/partitionKey'
        ]
        version: 2
      }
    }
  }
}

resource edgeContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-04-15' = {
  parent: database
  name: edgeContainerName
  properties: {
    resource: {
      id: edgeContainerName
      partitionKey: {
        kind: 'Hash'
        paths: [
          '/partitionKey'
        ]
        version: 2
      }
    }
  }
}

resource developerDataAccess 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2025-04-15' = {
  parent: account
  name: guid(account.id, developerPrincipalId, dataContributorRoleDefinitionId)
  properties: {
    principalId: developerPrincipalId
    roleDefinitionId: dataContributorRoleDefinitionId
    scope: account.id
  }
}

output databaseName string = database.name
output vertexContainerName string = vertexContainer.name
output edgeContainerName string = edgeContainer.name
