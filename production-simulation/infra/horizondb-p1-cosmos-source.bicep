targetScope = 'resourceGroup'

@minLength(7)
@maxLength(31)
param runId string
param leaseId string
param runnerSubnetId string
param cosmosPrivateDnsZoneId string

var accountName = 'afhsp1${runId}'
var tags = {
  application: 'agefreighter'
  campaign: 'afhs-20261118'
  phase: 'p1'
  qualificationLease: leaseId
}

resource account 'Microsoft.DocumentDB/databaseAccounts@2025-04-15' = {
  name: accountName
  location: 'japaneast'
  kind: 'GlobalDocumentDB'
  tags: tags
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    disableLocalAuth: true
    enableAutomaticFailover: false
    enableMultipleWriteLocations: false
    locations: [
      {
        locationName: 'japaneast'
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    minimalTlsVersion: 'Tls12'
    networkAclBypass: 'None'
    publicNetworkAccess: 'Disabled'
  }
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2025-04-15' = {
  parent: account
  name: 'agefreighter'
  properties: {
    resource: { id: 'agefreighter' }
    options: { throughput: 60000 }
  }
}

resource containers 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-04-15' = [for dataset in ['base', 'append', 'upsert']: {
  parent: database
  name: dataset
  properties: {
    resource: {
      id: dataset
      partitionKey: {
        paths: ['/partitionKey']
        kind: 'Hash'
        version: 2
      }
    }
    options: {}
  }
}]

resource privateEndpoint 'Microsoft.Network/privateEndpoints@2024-07-01' = {
  name: '${accountName}-pe'
  location: 'japaneast'
  tags: tags
  properties: {
    subnet: { id: runnerSubnetId }
    privateLinkServiceConnections: [
      {
        name: 'cosmos'
        properties: {
          privateLinkServiceId: account.id
          groupIds: ['Sql']
        }
      }
    ]
  }
}

resource privateDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = {
  parent: privateEndpoint
  name: 'cosmos'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'cosmos'
        properties: { privateDnsZoneId: cosmosPrivateDnsZoneId }
      }
    ]
  }
}

output accountId string = account.id
