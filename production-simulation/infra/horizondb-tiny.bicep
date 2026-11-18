targetScope = 'resourceGroup'

@description('Run-scoped prefix for the approved tiny qualification only.')
@minLength(3)
@maxLength(30)
param prefix string

param location string = 'japaneast'

param administratorUsername string = 'afrunner'

@secure()
param sshPublicKey string

@description('Verified Ubuntu 24.04 image version.')
param imageVersion string

@description('UTC daily emergency shutdown time, HHmm.')
param shutdownTime string

@description('Enable only on a second deployment after pools finish initial creation.')
param configureAGE bool = false

@description('Runner principal ID from the first deployment; empty skips administrator creation.')
param administratorObjectId string = ''

var versions = [
  '17'
  '18'
]
var tags = {
  application: 'agefreighter'
  purpose: 'horizondb-qualification'
  phase: 'tiny'
  run: prefix
}
var cosmosName = '${replace(prefix, '-', '')}${uniqueString(resourceGroup().id)}'
var dnsNames = [
  'privatelink.horizondb.azure.com'
  'privatelink.documents.azure.com'
]

resource outboundIP 'Microsoft.Network/publicIPAddresses@2024-07-01' = {
  name: '${prefix}-outbound'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

resource nat 'Microsoft.Network/natGateways@2024-07-01' = {
  name: '${prefix}-nat'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIpAddresses: [
      { id: outboundIP.id }
    ]
    idleTimeoutInMinutes: 10
  }
}

resource nsg 'Microsoft.Network/networkSecurityGroups@2024-07-01' = {
  name: '${prefix}-nsg'
  location: location
  tags: tags
  properties: {
    securityRules: [
      {
        name: 'deny-inbound'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2024-07-01' = {
  name: '${prefix}-vnet'
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.86.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'tiny'
        properties: {
          addressPrefix: '10.86.1.0/24'
          defaultOutboundAccess: false
          privateEndpointNetworkPolicies: 'Disabled'
          natGateway: { id: nat.id }
          networkSecurityGroup: { id: nsg.id }
        }
      }
    ]
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2024-07-01' = {
  name: '${prefix}-nic'
  location: location
  tags: tags
  properties: {
    enableAcceleratedNetworking: true
    ipConfigurations: [
      {
        name: 'private'
        properties: {
          privateIPAllocationMethod: 'Dynamic'
          subnet: { id: vnet.properties.subnets[0].id }
        }
      }
    ]
  }
}

resource runner 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: '${prefix}-runner'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    hardwareProfile: { vmSize: 'Standard_D8ds_v5' }
    securityProfile: {
      securityType: 'TrustedLaunch'
      uefiSettings: {
        secureBootEnabled: true
        vTpmEnabled: true
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: imageVersion
      }
      osDisk: {
        createOption: 'FromImage'
        diskSizeGB: 128
        managedDisk: { storageAccountType: 'StandardSSD_LRS' }
        deleteOption: 'Delete'
      }
    }
    osProfile: {
      computerName: 'horizon-tiny'
      adminUsername: administratorUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        provisionVMAgent: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${administratorUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        { id: nic.id }
      ]
    }
  }
}

resource shutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = {
  name: 'shutdown-computevm-${runner.name}'
  location: location
  tags: tags
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    targetResourceId: runner.id
    timeZoneId: 'UTC'
    dailyRecurrence: { time: shutdownTime }
    notificationSettings: { status: 'Disabled' }
  }
}

resource clusters 'Microsoft.HorizonDb/clusters@2026-10-01-preview' = [for version in versions: {
  name: '${prefix}-pg${version}'
  location: location
  tags: tags
  properties: {
    postgreSqlVersion: version
    network: { publicAccess: 'Disabled' }
    backup: {
      geoRedundancy: 'Disabled'
      retentionDays: 7
    }
  }
}]

resource ageParameters 'Microsoft.HorizonDb/parameterGroups@2026-10-01-preview' = [for version in versions: {
  name: '${prefix}-pg${version}-age'
  location: location
  tags: tags
  properties: {
    postgreSqlVersion: version
    description: 'Approved tiny qualification; initial managed preload defaults were empty.'
    parameters: [
      { name: 'azure.extensions', value: 'age' }
      { name: 'shared_preload_libraries', value: 'age' }
    ]
  }
}]

resource pools 'Microsoft.HorizonDb/clusters/pools@2026-10-01-preview' = [for (version, i) in versions: {
  name: '${clusters[i].name}/writer'
  tags: tags
  properties: {
    computeModel: {
      type: 'Provisioned'
      computeProcessor: 'Intel'
      computeGeneration: 1
      vCores: 2
    }
    nodeCount: 1
    highAvailability: { mode: 'Disabled' }
    parameterGroup: {
      parameterGroupResourceId: configureAGE
        ? ageParameters[i].id
        : subscriptionResourceId('Microsoft.HorizonDb/locations/parameterGroups', location, 'default_pg${version}')
      applyMethod: 'Immediate'
    }
  }
}]

resource authentications 'Microsoft.HorizonDb/clusters/authentications@2026-10-01-preview' = [for (version, i) in versions: {
  name: '${clusters[i].name}/microsoftEntra'
  properties: { state: 'Enabled' }
  dependsOn: [pools[i]]
}]

resource administrators 'Microsoft.HorizonDb/clusters/microsoftEntraAdministrators@2026-10-01-preview' = [for (version, i) in versions: if (!empty(administratorObjectId)) {
  name: '${clusters[i].name}/${administratorObjectId}'
  properties: {
    principalName: runner.name
    principalType: 'ServicePrincipal'
    tenantId: tenant().tenantId
  }
  dependsOn: [authentications[i]]
}]

resource deadlineRoles 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (version, i) in versions: {
  scope: clusters[i]
  name: guid(clusters[i].id, runner.id, 'tiny-compute-deadline')
  properties: {
    principalId: runner.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
  }
}]

module cosmos '../../infra/modules/cosmos.bicep' = {
  name: 'tiny-cosmos'
  params: {
    accountName: cosmosName
    location: location
    publicNetworkAccess: 'Disabled'
    tags: tags
  }
}

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2025-04-15' existing = {
  name: cosmosName
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2025-04-15' = {
  parent: cosmosAccount
  name: 'agefreighter'
  properties: {
    resource: { id: 'agefreighter' }
  }
  dependsOn: [cosmos]
}

resource containers 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2025-04-15' = [for name in ['vertices', 'edges', 'tiny-explicit', 'tiny-gremlin']: {
  parent: database
  name: name
  properties: {
    resource: {
      id: name
      partitionKey: {
        paths: ['/partitionKey']
        kind: 'Hash'
        version: 2
      }
    }
  }
}]

resource cosmosRole 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2025-04-15' = {
  parent: cosmosAccount
  name: guid(cosmosAccount.id, runner.id, 'tiny-data')
  properties: {
    principalId: runner.identity.principalId
    roleDefinitionId: '${cosmosAccount.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    scope: cosmosAccount.id
  }
  dependsOn: [cosmos]
}

resource zones 'Microsoft.Network/privateDnsZones@2024-06-01' = [for name in dnsNames: {
  name: name
  location: 'global'
  tags: tags
}]

resource links 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = [for (name, i) in dnsNames: {
  name: '${name}/tiny-vnet'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: vnet.id }
  }
  dependsOn: [zones[i]]
}]

resource horizonEndpoints 'Microsoft.Network/privateEndpoints@2024-07-01' = [for (version, i) in versions: {
  name: '${prefix}-pg${version}-pe'
  location: location
  tags: tags
  properties: {
    subnet: { id: vnet.properties.subnets[0].id }
    privateLinkServiceConnections: [
      {
        name: 'writer'
        properties: {
          privateLinkServiceId: clusters[i].id
          groupIds: [clusters[i].name]
        }
      }
    ]
  }
  dependsOn: [pools[i]]
}]

resource horizonDNS 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = [for (version, i) in versions: {
  name: '${horizonEndpoints[i].name}/horizon'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'horizon'
        properties: { privateDnsZoneId: zones[0].id }
      }
    ]
  }
}]

resource cosmosEndpoint 'Microsoft.Network/privateEndpoints@2024-07-01' = {
  name: '${prefix}-cosmos-pe'
  location: location
  tags: tags
  properties: {
    subnet: { id: vnet.properties.subnets[0].id }
    privateLinkServiceConnections: [
      {
        name: 'cosmos'
        properties: {
          privateLinkServiceId: cosmos.outputs.accountId
          groupIds: ['Sql']
        }
      }
    ]
  }
}

resource cosmosDNS 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = {
  parent: cosmosEndpoint
  name: 'cosmos'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'cosmos'
        properties: { privateDnsZoneId: zones[1].id }
      }
    ]
  }
}

output runnerName string = runner.name
output runnerPrincipalId string = runner.identity.principalId
output clusterNames array = [for (version, i) in versions: clusters[i].name]
output cosmosEndpointURL string = cosmos.outputs.endpoint
