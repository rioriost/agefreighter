targetScope = 'resourceGroup'

@description('Evidence and the independent watchdog must live outside the disposable scale resource group.')
param prefix string
param location string = 'japaneast'
param existingVnetName string
param existingSubnetName string = 'tiny'
param evidenceWriterPrincipalId string
@description('Protect the new evidence account against accidental ARM deletion, including deletion of its group.')
param protectEvidenceAccount bool = false

var tags = {
  application: 'agefreighter'
  purpose: 'horizondb-scale-safety'
  campaign: prefix
  retain: 'evidence'
}

resource vnet 'Microsoft.Network/virtualNetworks@2024-07-01' existing = {
  name: existingVnetName
}
resource subnet 'Microsoft.Network/virtualNetworks/subnets@2024-07-01' existing = {
  parent: vnet
  name: existingSubnetName
}
resource evidence 'Microsoft.Storage/storageAccounts@2026-09-01' = {
  name: 'afhe${uniqueString(resourceGroup().id, prefix)}'
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Disabled'
    networkAcls: { bypass: 'None', defaultAction: 'Deny' }
  }
}
resource blobs 'Microsoft.Storage/storageAccounts/blobServices@2026-09-01' = {
  parent: evidence
  name: 'default'
  properties: {
    isVersioningEnabled: true
    deleteRetentionPolicy: { enabled: true, days: 7 }
    containerDeleteRetentionPolicy: { enabled: true, days: 7 }
  }
}
resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2026-09-01' = {
  parent: blobs
  name: 'qualification'
  properties: { publicAccess: 'None' }
}
resource writer 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(container.id, evidenceWriterPrincipalId, 'evidence-writer')
  scope: container
  properties: {
    principalId: evidenceWriterPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}
resource zone 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: 'privatelink.blob.${environment().suffixes.storage}'
  location: 'global'
  tags: tags
}
resource link 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: zone
  name: '${prefix}-evidence'
  location: 'global'
  properties: { registrationEnabled: false, virtualNetwork: { id: vnet.id } }
}
resource endpoint 'Microsoft.Network/privateEndpoints@2024-07-01' = {
  name: '${prefix}-evidence'
  location: location
  tags: tags
  properties: {
    subnet: { id: subnet.id }
    privateLinkServiceConnections: [
      {
        name: 'blob'
        properties: { privateLinkServiceId: evidence.id, groupIds: ['blob'] }
      }
    ]
  }
}
resource dns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-07-01' = {
  parent: endpoint
  name: 'default'
  properties: { privateDnsZoneConfigs: [{ name: 'blob', properties: { privateDnsZoneId: zone.id } }] }
}
resource evidenceLock 'Microsoft.Authorization/locks@2016-09-01' = if (protectEvidenceAccount) {
  name: 'retain-qualification-evidence'
  scope: evidence
  properties: {
    level: 'CanNotDelete'
    notes: 'Remove deliberately only after qualification evidence is preserved elsewhere.'
  }
}
resource automation 'Microsoft.Automation/automationAccounts@2024-10-23' = {
  name: '${prefix}-guard'
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: { sku: { name: 'Basic' }, disableLocalAuth: true }
}
resource snapshotRole 'Microsoft.Authorization/roleDefinitions@2022-04-01' = {
  name: guid(resourceGroup().id, prefix, 'preserve-evidence')
  properties: {
    roleName: '${prefix}-preserve-evidence'
    description: 'Preserve worker OS-disk evidence outside cleanup scope; no evidence or existing resource deletion.'
    type: 'CustomRole'
    assignableScopes: [resourceGroup().id]
    permissions: [
      {
        actions: [
          'Microsoft.Compute/disks/read'
          'Microsoft.Compute/snapshots/read'
          'Microsoft.Compute/snapshots/write'
          'Microsoft.Resources/subscriptions/resourceGroups/read'
        ]
        notActions: []
      }
    ]
  }
}
resource snapshotWriter 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, automation.id, 'preserve-evidence')
  properties: {
    principalId: automation.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: snapshotRole.id
  }
}
resource runtime 'Microsoft.Automation/automationAccounts/runtimeEnvironments@2024-10-23' = {
  parent: automation
  name: 'qualification-python310'
  location: location
  properties: { runtime: { language: 'Python', version: '3.10' } }
}
resource runbook 'Microsoft.Automation/automationAccounts/runbooks@2024-10-23' = {
  parent: automation
  name: 'EnforceQualificationLease'
  location: location
  tags: tags
  properties: {
    runbookType: 'Python'
    runtimeEnvironment: runtime.name
    draft: {}
    logProgress: false
    logVerbose: false
    description: 'Exact-scope, evidence-preserving deadline and conservative cost enforcement.'
  }
}

output evidenceAccountName string = evidence.name
output evidenceContainerId string = container.id
output guardAccountName string = automation.name
output guardPrincipalId string = automation.identity.principalId
