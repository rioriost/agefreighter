targetScope = 'resourceGroup'

@description('Azure region of the existing HorizonDB and VNet resources.')
param location string

@description('Resource ID of the existing HorizonDB cluster.')
param clusterId string

@description('Name of the existing production-simulation VNet.')
param virtualNetworkName string

@description('Name of the subnet that hosts the private endpoint.')
param subnetName string = 'loader'

@description('Private Link group ID read from the deployed HorizonDB privateLinkResources endpoint.')
param groupId string

@description('Private DNS zone read from the deployed HorizonDB privateLinkResources endpoint.')
param privateDnsZoneName string

@description('Private endpoint resource name.')
param privateEndpointName string

@description('Tags applied to private networking resources.')
param tags object = {
  application: 'agefreighter'
  purpose: 'horizondb-p2-feasibility'
  managedBy: 'bicep'
}

resource virtualNetwork 'Microsoft.Network/virtualNetworks@2024-07-01' existing = {
  name: virtualNetworkName
}

resource subnet 'Microsoft.Network/virtualNetworks/subnets@2024-07-01' existing = {
  parent: virtualNetwork
  name: subnetName
}

resource privateDNS 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: privateDnsZoneName
  location: 'global'
  tags: tags
}

resource privateDNSLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: privateDNS
  name: '${privateEndpointName}-link'
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}

resource privateEndpoint 'Microsoft.Network/privateEndpoints@2024-07-01' = {
  name: privateEndpointName
  location: location
  tags: tags
  properties: {
    subnet: {
      id: subnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'horizondb'
        properties: {
          privateLinkServiceId: clusterId
          groupIds: [
            groupId
          ]
        }
      }
    ]
  }
}

resource privateDNSGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = {
  parent: privateEndpoint
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'horizondb'
        properties: {
          privateDnsZoneId: privateDNS.id
        }
      }
    ]
  }
  dependsOn: [
    privateDNSLink
  ]
}

output privateEndpointId string = privateEndpoint.id
output privateDnsZoneId string = privateDNS.id

