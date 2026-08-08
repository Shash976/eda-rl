// infra/main.bicep — the eda-rl dashboard, on Azure Container Apps.
//
// Resource-group-scoped template: run `az group create` first (see
// infra/README.md for the full manual deploy runbook — deployment is a
// deliberate manual step for this project, not wired into CI).
//
// Resources: an Azure Container Registry (the image lives here — no GHCR
// dependency), a Log Analytics workspace (required by the Container Apps
// Environment for logs), the Environment itself, and one Container App that
// serves both the FastAPI backend and the built React frontend from a single
// image/process. The Container App pulls from ACR via its own system-assigned
// managed identity + an AcrPull role assignment — no admin credentials or
// stored registry secrets.

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Base name used to derive resource names (Container App, environment, workspace).')
@minLength(3)
@maxLength(24)
param appName string = 'eda-rl-dashboard'

@description('Azure Container Registry name — must be globally unique, alphanumeric only.')
@minLength(5)
@maxLength(50)
param acrName string

@description('Image tag to deploy (the tag pushed by `az acr build`, e.g. "latest" or a git SHA).')
param containerImageTag string = 'latest'

@description('vCPU allotted to the container — demo-sized on purpose.')
param containerCpu string = '0.25'

@description('Memory allotted to the container — demo-sized on purpose.')
param containerMemory string = '0.5Gi'

var imageName = 'eda-rl-api'
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d' // built-in "AcrPull" role

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false // pull is via managed identity, not admin creds
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30 // short retention — cost-conscious for a demo
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: imageName
          image: '${acr.properties.loginServer}/${imageName}:${containerImageTag}'
          resources: {
            cpu: json(containerCpu)
            memory: containerMemory
          }
        }
      ]
      scale: {
        minReplicas: 0 // scale-to-zero: the deliberate cost trade-off for a
        maxReplicas: 1 // low-traffic demo dashboard (cold-start on first hit)
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

// Grant the Container App's own managed identity pull access to the ACR —
// no admin user, no stored registry secret.
resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, containerApp.id, 'AcrPull')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Public URL of the deployed dashboard.')
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn

@description('ACR login server, for `az acr build`/`docker push`.')
output acrLoginServer string = acr.properties.loginServer
