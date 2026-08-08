# Deploying the eda-rl dashboard to Azure Container Apps

`main.bicep` provisions everything the dashboard needs: an Azure Container
Registry, a Log Analytics workspace, a Container Apps Environment, and one
Container App (FastAPI backend + built React frontend, one image, one
process). The Container App pulls from ACR via its own system-assigned
managed identity — no admin credentials, no stored registry secret.

**Deployment is manual, on purpose.** There is no CI workflow that deploys
automatically on push — see the top-level `AGENTS.md`/repo history for why.
Everything below is a command you run yourself, once you have:

- an Azure subscription and `az login` already done,
- the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli),
- Docker (for the image build step, run remotely by `az acr build` — no
  local Docker daemon needed for that specific step, but it's used here for
  local verification first).

## 1. Validate the template (free, no Azure resources created)

```bash
az bicep build --file infra/main.bicep
```

This is also what CI's `infra-lint` job runs on every PR.

## 2. Create the resource group (one time)

```bash
az group create --name eda-rl-dashboard-rg --location <region>   # e.g. eastus
```

## 3. Provision the infrastructure

`acrName` must be globally unique (Azure Container Registry names are a
global DNS-ish namespace) — pick something like `edarldash<yourinitials>`.

```bash
az deployment group create \
  --resource-group eda-rl-dashboard-rg \
  --template-file infra/main.bicep \
  --parameters appName=eda-rl-dashboard acrName=<globally-unique-name>
```

The Container App is created here with **no image yet** — `containerImageTag`
defaults to `latest`, which doesn't exist in a brand-new registry, so the
first revision will fail to start. That's expected; step 4 fixes it.

Optional dry-run first (no resources created, no cost):

```bash
az deployment group validate \
  --resource-group eda-rl-dashboard-rg \
  --template-file infra/main.bicep \
  --parameters appName=eda-rl-dashboard acrName=<globally-unique-name>
```

## 4. Build and push the image

`az acr build` builds in the cloud — no local Docker daemon required, and it
authenticates to the registry automatically:

```bash
az acr build --registry <globally-unique-name> --image eda-rl-api:latest \
  -f Dockerfile.api .
```

(Building with BuildKit locally first, per the root `AGENTS.md`, is still
the fast local dev loop: `DOCKER_BUILDKIT=1 docker build -f Dockerfile.api
-t eda-rl-api:local .` — `az acr build` is the one that actually needs to
land in ACR for the Container App to pull.)

## 5. Point the Container App at the new image

```bash
az containerapp update \
  --name eda-rl-dashboard \
  --resource-group eda-rl-dashboard-rg \
  --image <globally-unique-name>.azurecr.io/eda-rl-api:latest
```

## 6. Get the URL

```bash
az deployment group show \
  --resource-group eda-rl-dashboard-rg \
  --name main \
  --query properties.outputs.containerAppFqdn.value -o tsv
```

(Same value as the `containerAppFqdn` Bicep output from step 3.)

## Redeploying after a code change

Repeat steps 4 and 5 with a new tag (e.g. the git SHA) — Bicep/ARM
deployments are idempotent, so re-running step 3 is also safe if
infrastructure itself changed (Container App CPU/memory, ingress, etc.).

## Cost notes

- The Container App scales to zero (`minReplicas: 0`) — no compute cost
  while idle, at the price of a cold start on the first request after a
  quiet period.
- ACR Basic SKU and Log Analytics (30-day retention) are the two
  always-on costs; both are the smallest tier that still works for a
  single low-traffic app.
- Tear everything down with `az group delete --name eda-rl-dashboard-rg`
  when you're done — this deletes every resource this template created.

az acr build --registry edarldashboardapp --image eda-rl-api:latest -f Dockerfile.api .
2. Point the Container App at it: az containerapp update --name eda-rl-dashboard --resource-group eda-rl-dashboard-rg --image edarldashboardapp.azurecr.io/eda-rl-api:latest

az containerapp show --name eda-rl-dashboard --resource-group eda-rl-dashboard-rg --query "properties.provisioningState" -o tsv
1. Poll that (don't spam update again) until it's Succeeded or Failed.