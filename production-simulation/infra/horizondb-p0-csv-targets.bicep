targetScope = 'resourceGroup'

@allowed(['japaneast'])
param location string = 'japaneast'
param leaseId string
param configureAGE bool = false

@description('The single independently verified, operator-approved runner NAT IPv4; no CIDR or range.')
param approvedRunnerNatIPv4 string

@description('Leave empty on initial deployment; enroll the runner only after the writer and Entra auth are ready.')
param administratorObjectId string = ''

@allowed(['csv', 'postgresql', 'neo4j', 'cosmos'])
param source string = 'csv'

var versions = ['17', '18']
var prefix = 'afhs-20261118'
var tags = {
  application: 'agefreighter'
  campaign: prefix
  qualificationLease: leaseId
  phase: 'p0'
  source: source
}

resource clusters 'Microsoft.HorizonDb/clusters@2026-10-01-preview' = [for version in versions: {
  name: '${prefix}-pg${version}'
  location: location
  tags: tags
  properties: {
    postgreSqlVersion: version
    network: { publicAccess: 'Enabled' }
    backup: { geoRedundancy: 'Disabled', retentionDays: 7 }
  }
}]

resource parameters 'Microsoft.HorizonDb/parameterGroups@2026-10-01-preview' = [for version in versions: {
  name: '${prefix}-pg${version}-age'
  location: location
  tags: tags
  properties: {
    postgreSqlVersion: version
    description: 'Bounded approved P0 ${source} mode qualification.'
    parameters: [
      { name: 'azure.extensions', value: 'age' }
      { name: 'shared_preload_libraries', value: 'age' }
    ]
  }
}]

resource pools 'Microsoft.HorizonDb/clusters/pools@2026-10-01-preview' = [for (version, i) in versions: {
  name: '${clusters[i].name}/writer'
  properties: {
    computeModel: {
      type: 'Provisioned'
      computeProcessor: 'Intel'
      computeGeneration: 1
      vCores: 8
    }
    nodeCount: 1
    highAvailability: { mode: 'Disabled' }
    parameterGroup: {
      parameterGroupResourceId: configureAGE
        ? parameters[i].id
        : subscriptionResourceId('Microsoft.HorizonDb/locations/parameterGroups', location, 'default_pg${version}')
      applyMethod: 'Immediate'
    }
  }
}]

resource firewall 'Microsoft.HorizonDb/clusters/pools/firewallRules@2026-10-01-preview' = [for (version, i) in versions: {
  name: '${pools[i].name}/approved-runner-nat-only'
  properties: {
    startIpAddress: approvedRunnerNatIPv4
    endIpAddress: approvedRunnerNatIPv4
  }
}]

resource authentication 'Microsoft.HorizonDb/clusters/authentications@2026-10-01-preview' = [for (version, i) in versions: {
  name: '${clusters[i].name}/microsoftEntra'
  properties: { state: 'Enabled' }
  dependsOn: [pools[i]]
}]

resource administrators 'Microsoft.HorizonDb/clusters/microsoftEntraAdministrators@2026-10-01-preview' = [for (version, i) in versions: if (!empty(administratorObjectId)) {
  name: '${clusters[i].name}/${administratorObjectId}'
  properties: {
    principalName: 'afht-20261118-runner'
    principalType: 'ServicePrincipal'
    tenantId: tenant().tenantId
  }
  dependsOn: [authentication[i]]
}]
