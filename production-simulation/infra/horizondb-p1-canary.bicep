targetScope = 'resourceGroup'

param leaseId string
@secure()
param sshPublicKey string
param fallbackShutdownTime string
param administratorUsername string

var location = 'japaneast'
var name = 'afhs-20261118-p1-canary'
var tags = {
  application: 'agefreighter'
  campaign: 'afhs-20261118'
  qualificationLease: leaseId
  phase: 'p1'
  purpose: 'lifecycle-canary'
}

resource vnet 'Microsoft.Network/virtualNetworks@2024-07-01' existing = {
  scope: resourceGroup('rg-agefreighter-horizon-tiny-20261118')
  name: 'afht-20261118-vnet'
}
resource subnet 'Microsoft.Network/virtualNetworks/subnets@2024-07-01' existing = {
  parent: vnet
  name: 'tiny'
}
resource nic 'Microsoft.Network/networkInterfaces@2024-07-01' = {
  name: '${name}-nic'
  location: location
  tags: tags
  properties: {
    enableAcceleratedNetworking: true
    ipConfigurations: [
      {
        name: 'private'
        properties: {
          privateIPAllocationMethod: 'Dynamic'
          subnet: { id: subnet.id }
        }
      }
    ]
  }
}
resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: name
  location: location
  tags: tags
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
        version: '24.04.202610250'
      }
      osDisk: {
        name: '${name}-os'
        createOption: 'FromImage'
        diskSizeGB: 128
        managedDisk: { storageAccountType: 'Premium_LRS' }
        caching: 'ReadWrite'
        deleteOption: 'Delete'
      }
    }
    osProfile: {
      computerName: name
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
  name: 'shutdown-computevm-${vm.name}'
  location: location
  tags: tags
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    targetResourceId: vm.id
    timeZoneId: 'UTC'
    dailyRecurrence: { time: fallbackShutdownTime }
    notificationSettings: { status: 'Disabled' }
  }
}
