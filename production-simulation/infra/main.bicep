targetScope = 'resourceGroup'

@minLength(3)
@maxLength(20)
@description('Short environment name used only for production simulation resources.')
param environmentName string

@description('Azure region selected during review.')
param location string

@allowed([
  '1'
  '2'
  '3'
])
@description('One availability zone shared by every zonal resource.')
param availabilityZone string

@allowed([
  'p0'
  'p1'
  'p2'
  'p3'
])
@description('Approved stage represented by this deployment.')
param testPhase string

@description('Linux administrator name for the three private VMs.')
param administratorUsername string

@secure()
@description('Reviewed SSH public key. Supply at deployment time; never commit it.')
param sshPublicKey string

@secure()
@description('PostgreSQL administrator password. Supply from an approved secret source.')
param postgresAdministratorPassword string

@secure()
@description('Neo4j password stored in the P0 Key Vault. Supply from an approved secret source.')
param neo4jPassword string

@description('Neo4j VM SKU. Availability in the selected zone is a review gate.')
param neo4jVmSize string = 'Standard_E32bds_v5'

@description('agefreighter VM SKU. Availability in the selected zone is a review gate.')
param loaderVmSize string = 'Standard_D16ds_v5'

@description('Flexible Server SKU name in the Memory Optimized tier.')
param postgresSkuName string = 'Standard_E32ds_v5'

@minValue(512)
@maxValue(65536)
@description('Neo4j durable data disk size in GiB.')
param neo4jDataDiskSizeGB int = 4096

@minValue(4000)
@maxValue(80000)
@description('Provisioned IOPS for each Neo4j Premium SSD v2 data disk.')
param neo4jDataDiskIOPS int = 40000

@minValue(500)
@maxValue(1200)
@description('Provisioned MiB/s for each Neo4j Premium SSD v2 data disk.')
param neo4jDataDiskMBps int = 1000

@minValue(128)
@maxValue(8192)
@description('Initial PostgreSQL storage size in GiB; P2 must approve any increase.')
param postgresStorageSizeGB int = 8192

@allowed([
  'Disabled'
  'SameZone'
])
@description('Must match the intended production durability model.')
param postgresHighAvailability string = 'SameZone'

var prefix = 'afps-${environmentName}'
var networkPrefix = '10.42.0.0/16'
var loaderSubnetPrefix = '10.42.1.0/27'
var neo4jSubnetPrefix = '10.42.2.0/24'
var postgresSubnetPrefix = '10.42.3.0/24'
var keyVaultSecretsUserRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4633458b-17de-408a-b874-0445c86b69e6'
)
var tags = {
  application: 'agefreighter'
  purpose: 'production-simulation'
  environment: environmentName
  phase: testPhase
  managedBy: 'bicep'
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-07-01' = {
  name: '${prefix}-vnet'
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        networkPrefix
      ]
    }
  }
}

resource loaderSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-07-01' = {
  parent: virtualNetwork
  name: 'loader'
  properties: {
    addressPrefix: loaderSubnetPrefix
    privateEndpointNetworkPolicies: 'Enabled'
  }
}

resource neo4jSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-07-01' = {
  parent: virtualNetwork
  name: 'neo4j'
  properties: {
    addressPrefix: neo4jSubnetPrefix
    networkSecurityGroup: {
      id: neo4jNetworkSecurityGroup.id
    }
    privateEndpointNetworkPolicies: 'Enabled'
  }
}

resource postgresSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-07-01' = {
  parent: virtualNetwork
  name: 'postgresql'
  properties: {
    addressPrefix: postgresSubnetPrefix
    delegations: [
      {
        name: 'postgresql-flexible-server'
        properties: {
          serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
        }
      }
    ]
    privateEndpointNetworkPolicies: 'Enabled'
  }
}

resource neo4jNetworkSecurityGroup 'Microsoft.Network/networkSecurityGroups@2024-07-01' = {
  name: '${prefix}-neo4j-nsg'
  location: location
  tags: tags
  properties: {
    securityRules: [
      {
        name: 'bolt-from-loader'
        properties: {
          priority: 100
          access: 'Allow'
          direction: 'Inbound'
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '7687'
          sourceAddressPrefix: loaderSubnetPrefix
          destinationAddressPrefix: '*'
        }
      }
      {
        name: 'deny-vnet-inbound'
        properties: {
          priority: 200
          access: 'Deny'
          direction: 'Inbound'
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

resource privateDNSZone 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'afps.internal'
  location: 'global'
  tags: tags
}

resource privateDNSLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: privateDNSZone
  name: '${prefix}-link'
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource loaderNIC 'Microsoft.Network/networkInterfaces@2024-07-01' = {
  name: '${prefix}-loader-nic'
  location: location
  tags: tags
  properties: {
    ipConfigurations: [
      {
        name: 'primary'
        properties: {
          privateIPAllocationMethod: 'Static'
          privateIPAddress: '10.42.1.10'
          subnet: {
            id: loaderSubnet.id
          }
        }
      }
    ]
  }
}

resource neo4j44NIC 'Microsoft.Network/networkInterfaces@2024-07-01' = {
  name: '${prefix}-neo4j44-nic'
  location: location
  tags: tags
  properties: {
    ipConfigurations: [
      {
        name: 'primary'
        properties: {
          privateIPAllocationMethod: 'Static'
          privateIPAddress: '10.42.2.44'
          subnet: {
            id: neo4jSubnet.id
          }
        }
      }
    ]
  }
}

resource neo4j526NIC 'Microsoft.Network/networkInterfaces@2024-07-01' = {
  name: '${prefix}-neo4j526-nic'
  location: location
  tags: tags
  properties: {
    ipConfigurations: [
      {
        name: 'primary'
        properties: {
          privateIPAllocationMethod: 'Static'
          privateIPAddress: '10.42.2.52'
          subnet: {
            id: neo4jSubnet.id
          }
        }
      }
    ]
  }
}

resource neo4j44DNS 'Microsoft.Network/privateDnsZones/A@2024-06-01' = {
  parent: privateDNSZone
  name: 'neo4j44'
  properties: {
    ttl: 300
    aRecords: [
      {
        ipv4Address: '10.42.2.44'
      }
    ]
  }
}

resource neo4j526DNS 'Microsoft.Network/privateDnsZones/A@2024-06-01' = {
  parent: privateDNSZone
  name: 'neo4j526'
  properties: {
    ttl: 300
    aRecords: [
      {
        ipv4Address: '10.42.2.52'
      }
    ]
  }
}

resource neo4j44DataDisk 'Microsoft.Compute/disks@2024-03-02' = {
  name: '${prefix}-neo4j44-data'
  location: location
  zones: [
    availabilityZone
  ]
  tags: tags
  sku: {
    name: 'PremiumV2_LRS'
  }
  properties: {
    creationData: {
      createOption: 'Empty'
    }
    diskSizeGB: neo4jDataDiskSizeGB
    diskIOPSReadWrite: neo4jDataDiskIOPS
    diskMBpsReadWrite: neo4jDataDiskMBps
    networkAccessPolicy: 'DenyAll'
    publicNetworkAccess: 'Disabled'
  }
}

resource neo4j526DataDisk 'Microsoft.Compute/disks@2024-03-02' = {
  name: '${prefix}-neo4j526-data'
  location: location
  zones: [
    availabilityZone
  ]
  tags: tags
  sku: {
    name: 'PremiumV2_LRS'
  }
  properties: {
    creationData: {
      createOption: 'Empty'
    }
    diskSizeGB: neo4jDataDiskSizeGB
    diskIOPSReadWrite: neo4jDataDiskIOPS
    diskMBpsReadWrite: neo4jDataDiskMBps
    networkAccessPolicy: 'DenyAll'
    publicNetworkAccess: 'Disabled'
  }
}

resource loaderVM 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: '${prefix}-loader'
  location: location
  zones: [
    availabilityZone
  ]
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: vmProperties(loaderVmSize, loaderNIC.id, null, administratorUsername, sshPublicKey)
}

resource neo4j44VM 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: '${prefix}-neo4j44'
  location: location
  zones: [
    availabilityZone
  ]
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: vmProperties(neo4jVmSize, neo4j44NIC.id, neo4j44DataDisk.id, administratorUsername, sshPublicKey)
}

resource neo4j526VM 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: '${prefix}-neo4j526'
  location: location
  zones: [
    availabilityZone
  ]
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: vmProperties(neo4jVmSize, neo4j526NIC.id, neo4j526DataDisk.id, administratorUsername, sshPublicKey)
}

resource postgresPrivateDNS 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: '${prefix}.private.postgres.database.azure.com'
  location: 'global'
  tags: tags
}

resource postgresPrivateDNSLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: postgresPrivateDNS
  name: '${prefix}-postgres-link'
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: take('${replace(prefix, '-', '')}pg${uniqueString(subscription().subscriptionId, environmentName)}', 63)
  location: location
  tags: tags
  sku: {
    name: postgresSkuName
    tier: 'MemoryOptimized'
  }
  properties: {
    administratorLogin: administratorUsername
    administratorLoginPassword: postgresAdministratorPassword
    version: '18'
    availabilityZone: availabilityZone
    createMode: 'Default'
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: postgresHighAvailability == 'SameZone' ? {
      mode: 'SameZone'
      standbyAvailabilityZone: availabilityZone
    } : {
      mode: 'Disabled'
    }
    network: {
      delegatedSubnetResourceId: postgresSubnet.id
      privateDnsZoneArmResourceId: postgresPrivateDNS.id
      publicNetworkAccess: 'Disabled'
    }
    storage: {
      autoGrow: 'Disabled'
      storageSizeGB: postgresStorageSizeGB
    }
  }
  dependsOn: [
    postgresPrivateDNSLink
  ]
}

resource postgresExtensions 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = {
  parent: postgres
  name: 'azure.extensions'
  properties: {
    source: 'user-override'
    value: 'AGE'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2024-11-01' = {
  name: take('${replace(prefix, '-', '')}kv${uniqueString(subscription().subscriptionId, environmentName)}', 24)
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Disabled'
    sku: {
      family: 'A'
      name: 'standard'
    }
    networkAcls: {
      bypass: 'None'
      defaultAction: 'Allow'
    }
  }
}

resource postgresPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2024-11-01' = {
  parent: keyVault
  name: 'postgres-admin-password'
  properties: {
    value: postgresAdministratorPassword
  }
}

resource neo4jPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2024-11-01' = {
  parent: keyVault
  name: 'neo4j-password'
  properties: {
    value: neo4jPassword
  }
}

resource keyVaultPrivateDNS 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'privatelink.vaultcore.azure.net'
  location: 'global'
  tags: tags
}

resource keyVaultPrivateDNSLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: keyVaultPrivateDNS
  name: '${prefix}-keyvault-link'
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource keyVaultPrivateEndpoint 'Microsoft.Network/privateEndpoints@2024-07-01' = {
  name: '${prefix}-keyvault-pe'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: loaderSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'keyvault'
        properties: {
          privateLinkServiceId: keyVault.id
          groupIds: [
            'vault'
          ]
        }
      }
    ]
  }
}

resource keyVaultPrivateDNSGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = {
  parent: keyVaultPrivateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'keyvault'
        properties: {
          privateDnsZoneId: keyVaultPrivateDNS.id
        }
      }
    ]
  }
}

resource loaderSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, loaderVM.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    principalId: loaderVM.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRoleId
  }
}

resource neo4j44SecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, neo4j44VM.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    principalId: neo4j44VM.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRoleId
  }
}

resource neo4j526SecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, neo4j526VM.id, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    principalId: neo4j526VM.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRoleId
  }
}

func vmProperties(vmSize string, nicID string, dataDiskID string?, adminUsername string, publicKey string) object => {
  hardwareProfile: {
    vmSize: vmSize
  }
  networkProfile: {
    networkInterfaces: [
      {
        id: nicID
        properties: {
          primary: true
          deleteOption: 'Detach'
        }
      }
    ]
  }
  osProfile: {
    computerName: take(replace('${prefix}-${uniqueString(nicID)}', '-', ''), 15)
    adminUsername: adminUsername
    linuxConfiguration: {
      disablePasswordAuthentication: true
      provisionVMAgent: true
      ssh: {
        publicKeys: [
          {
            path: '/home/${adminUsername}/.ssh/authorized_keys'
            keyData: publicKey
          }
        ]
      }
    }
  }
  storageProfile: {
    imageReference: {
      publisher: 'Canonical'
      offer: 'ubuntu-24_04-lts'
      sku: 'server'
      version: 'latest'
    }
    osDisk: {
      createOption: 'FromImage'
      deleteOption: 'Delete'
      diskSizeGB: 256
      managedDisk: {
        storageAccountType: 'Premium_LRS'
      }
    }
    dataDisks: dataDiskID == null ? [] : [
      {
        lun: 0
        createOption: 'Attach'
        deleteOption: 'Detach'
        managedDisk: {
          id: dataDiskID
        }
      }
    ]
  }
}

output resourceGroupName string = resourceGroup().name
output region string = location
output zone string = availabilityZone
output loaderVMName string = loaderVM.name
output neo4j44VMName string = neo4j44VM.name
output neo4j526VMName string = neo4j526VM.name
output postgresServerName string = postgres.name
output postgresFQDN string = postgres.properties.fullyQualifiedDomainName
output keyVaultName string = keyVault.name
