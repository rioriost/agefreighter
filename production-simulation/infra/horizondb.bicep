targetScope = 'resourceGroup'

@minLength(3)
@maxLength(63)
@description('Azure HorizonDB cluster name.')
param clusterName string

@minLength(3)
@maxLength(63)
@description('HorizonDB parameter group name.')
param parameterGroupName string

@description('Azure region selected during review.')
param location string

@allowed([
  17
  18
])
@description('PostgreSQL major version observed in the live HorizonDB capability response.')
param postgresMajor int = 18

@minValue(1)
@maxValue(96)
@description('vCores assigned to every HorizonDB compute replica.')
param vCores int = 8

@minValue(1)
@description('Total compute replica count, including the primary.')
param replicaCount int = 2

@allowed([
  'Intel'
  'AMD'
])
@description('Compute processor family. Intel is frozen for the primary comparison.')
param processorType string = 'Intel'

@allowed([
  'BestEffort'
  'Strict'
])
@description('Availability-zone placement policy for compute replicas.')
param zonePlacementPolicy string = 'Strict'

@description('PostgreSQL administrator login.')
param administratorLogin string

@secure()
@description('PostgreSQL administrator password. Supply at deployment time.')
param administratorLoginPassword string

@description('Tags applied to all HorizonDB resources.')
param tags object = {
  application: 'agefreighter'
  purpose: 'horizondb-p2-feasibility'
  managedBy: 'bicep'
}

resource ageParameterGroup 'Microsoft.HorizonDB/parameterGroups@2026-01-20-preview' = {
  name: parameterGroupName
  location: location
  tags: tags
  properties: {
    applyImmediately: true
    description: 'AGE preload for the agefreighter P2 feasibility study'
    parameters: [
      {
        name: 'azure.extensions'
        value: 'AGE'
      }
      {
        name: 'shared_preload_libraries'
        value: 'age'
      }
    ]
    pgVersion: postgresMajor
  }
}

resource horizon 'Microsoft.HorizonDB/clusters@2026-01-20-preview' = {
  name: clusterName
  location: location
  tags: tags
  properties: {
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorLoginPassword
    createMode: 'Create'
    parameterGroup: {
      applyImmediately: true
      id: ageParameterGroup.id
    }
    poolName: 'DefaultPool'
    processorType: processorType
    replicaCount: replicaCount
    vCores: vCores
    version: string(postgresMajor)
    zonePlacementPolicy: zonePlacementPolicy
  }
}

output clusterId string = horizon.id
output clusterName string = horizon.name
output fqdn string = horizon.properties.fullyQualifiedDomainName
output parameterGroupId string = ageParameterGroup.id
output postgresMajor int = postgresMajor
output replicaCount int = replicaCount
output vCores int = vCores

