targetScope = 'resourceGroup'

param leaseId string
@description('The single independently verified, operator-approved runner NAT IPv4; no CIDR or range.')
param approvedRunnerNatIPv4 string

@allowed(['17', '18'])
@description('Allocate exactly one PostgreSQL writer. The read-only renewal probe uses the default PG17.')
param postgresMajor string = '17'

var location = 'japaneast'
var name = 'afhs-20261118-pg${postgresMajor}'
var tags = {
  application: 'agefreighter'
  campaign: 'afhs-20261118'
  qualificationLease: leaseId
  phase: 'p1'
  source: 'csv'
}

resource cluster 'Microsoft.HorizonDb/clusters@2026-10-01-preview' = {
  name: name
  location: location
  tags: tags
  properties: {
    postgreSqlVersion: postgresMajor
    network: { publicAccess: 'Enabled' }
    backup: { geoRedundancy: 'Disabled', retentionDays: 7 }
  }
}

resource writer 'Microsoft.HorizonDb/clusters/pools@2026-10-01-preview' = {
  parent: cluster
  name: 'writer'
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
      parameterGroupResourceId: subscriptionResourceId('Microsoft.HorizonDb/locations/parameterGroups', location, 'default_pg${postgresMajor}')
      applyMethod: 'Immediate'
    }
  }
}

resource firewall 'Microsoft.HorizonDb/clusters/pools/firewallRules@2026-10-01-preview' = {
  parent: writer
  name: 'approved-runner-nat-only'
  properties: {
    startIpAddress: approvedRunnerNatIPv4
    endIpAddress: approvedRunnerNatIPv4
  }
}

resource auth 'Microsoft.HorizonDb/clusters/authentications@2026-10-01-preview' = {
  parent: cluster
  name: 'microsoftEntra'
  properties: { state: 'Enabled' }
  dependsOn: [writer]
}
