@description('Resource ID of the Cosmos DB account protected by the perimeter.')
param accountId string

@description('Public IPv4 address allowed to reach the protected account.')
param developerIp string

@description('Azure region for the network security perimeter.')
param location string

@description('Network security perimeter name.')
param perimeterName string

@description('Tags applied to the network security perimeter.')
param tags object

resource perimeter 'Microsoft.Network/networkSecurityPerimeters@2025-07-01' = {
  name: perimeterName
  location: location
  tags: tags
  properties: {}
}

resource profile 'Microsoft.Network/networkSecurityPerimeters/profiles@2025-07-01' = {
  parent: perimeter
  name: 'cosmos-local-development'
  properties: {}
}

resource developerIngress 'Microsoft.Network/networkSecurityPerimeters/profiles/accessRules@2025-07-01' = {
  parent: profile
  name: 'developer-ipv4'
  properties: {
    addressPrefixes: [
      '${developerIp}/32'
    ]
    direction: 'Inbound'
  }
}

resource cosmosAssociation 'Microsoft.Network/networkSecurityPerimeters/resourceAssociations@2025-07-01' = {
  parent: perimeter
  name: 'cosmos-account'
  properties: {
    accessMode: 'Enforced'
    privateLinkResource: {
      id: accountId
    }
    profile: {
      id: profile.id
    }
  }
}
