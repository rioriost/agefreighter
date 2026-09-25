import { createHash, randomUUID } from "node:crypto";
import { object, RunnerRecord, validateWhatIf } from "./runner";
import { existingGroupResources, RunnerControl } from "./runnerLifecycle";
import { reportStorageNames } from "./runnerReportStorage";

export interface StorageDeployment {
  phase: "previewed" | "submitted" | "unknown" | "ready" | "failed";
  id: string; principalId: string; roleId: string; template: Record<string, unknown>; hash: string; expiresAt: string;
  networkAccess?: string;
}
const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const digest = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const contributor = "ba92f5b4-2d11-453d-a403-e96b0029c9fe";

export function storageDraft(record: RunnerRecord, principalId: string): StorageDeployment {
  if (!uuid.test(principalId)) throw new Error("The signed-in Azure user object ID is unavailable.");
  const names = reportStorageNames(record), role = randomUUID();
  const roleId = `${names.id}/providers/Microsoft.Authorization/roleAssignments/${role}`;
  const template = {
    $schema: "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#", contentVersion: "1.0.0.0",
    resources: [
      { type: "Microsoft.Storage/storageAccounts", apiVersion: "2023-05-01", name: names.account, location: record.input.region,
        kind: "StorageV2", sku: { name: "Standard_LRS" }, tags: { application: "agefreighter", workflow: record.id, purpose: "artifact-transfer" },
        properties: { supportsHttpsTrafficOnly: true, minimumTlsVersion: "TLS1_2", allowBlobPublicAccess: false, allowSharedKeyAccess: false,
          publicNetworkAccess: "Enabled", networkAcls: { defaultAction: "Allow", bypass: "None" }, accessTier: "Hot" } },
      { type: "Microsoft.Storage/storageAccounts/blobServices/containers", apiVersion: "2023-05-01", name: `${names.account}/default/${names.container}`,
        dependsOn: [names.id], properties: { publicAccess: "None", metadata: { agefreighterworkflow: record.id } } },
      { type: "Microsoft.Authorization/roleAssignments", apiVersion: "2022-04-01", name: role, scope: names.id, dependsOn: [names.id],
        properties: { principalId, principalType: "User", roleDefinitionId: `/subscriptions/${record.input.subscriptionId}/providers/Microsoft.Authorization/roleDefinitions/${contributor}` } }
    ]
  };
  const id = `${names.id.slice(0, names.id.indexOf("/providers/"))}/providers/Microsoft.Resources/deployments/${names.account}-transfer`;
  return { phase: "previewed", id, principalId, roleId, template, hash: digest({ id, principalId, roleId, template }), expiresAt: new Date(Date.now() + 900000).toISOString() };
}

function fresh(record: RunnerRecord): StorageDeployment {
  const d = record.storageDeployment;
  if (!d || d.phase !== "previewed" || !Number.isFinite(Date.parse(d.expiresAt)) || Date.parse(d.expiresAt) <= Date.now() ||
    d.hash !== digest({ id: d.id, principalId: d.principalId, roleId: d.roleId, template: d.template })) throw new Error("Storage approval expired or was already submitted.");
  return d;
}

export async function reviewStorage(control: RunnerControl, record: RunnerRecord): Promise<void> {
  const d = fresh(record), names = reportStorageNames(record);
  const existing = await existingGroupResources(control, record);
  for (const id of [names.id, names.containerId, d.roleId, d.id]) {
    const version = id === d.roleId ? "2022-04-01" : id === d.id ? "2022-09-01" : "2023-05-01";
    if ((await control.request(record.input.subscriptionId, `${id}?api-version=${version}`)).status !== 404) throw new Error("Transfer storage or role already exists. Reconcile the retained deployment; do not overwrite it.");
  }
  let response = await control.request(record.input.subscriptionId, `${d.id}/whatIf?api-version=2022-09-01`, "POST",
    { properties: { mode: "Incremental", template: d.template, whatIfSettings: { resultFormat: "ResourceIdOnly" } } });
  for (let attempt = 0; attempt < 30; attempt++) {
    const result = object(response.value);
    if (result.status === "Succeeded") { validateWhatIf(result, [names.id, names.containerId, d.roleId], existing); return; }
    if (["Failed", "Canceled"].includes(String(result.status)) || !response.poll) throw new Error("Storage what-if did not prove the exact new account, container and account-scoped user role.");
    await control.sleep(2000); const poll = response.poll;
    response = await control.request(record.input.subscriptionId, poll); response.poll ??= poll;
  }
  throw new Error("Storage what-if remains pending. No deployment was submitted.");
}

/** Caller holds a workflow lock and has approved cost/network/RBAC details. */
export async function submitStorage(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  await reviewStorage(control, record); const d = fresh(record);
  const next: RunnerRecord = { ...record, storageDeployment: { ...d, phase: "submitted" } };
  await control.persist(next);
  try {
    const result = await control.request(record.input.subscriptionId, `${d.id}?api-version=2022-09-01`, "PUT", { properties: { mode: "Incremental", template: d.template } });
    if (result.status < 200 || result.status >= 300) throw new Error();
  } catch { next.storageDeployment!.phase = "unknown"; await control.persist(next); }
  return next;
}

/** Read-only deployment and network reconciliation, no implicit replay or role repair. */
export async function refreshStorage(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const d = record.storageDeployment;
  if (!d || d.phase === "previewed") return record;
  const result = await control.request(record.input.subscriptionId, `${d.id}?api-version=2022-09-01`);
  const state = result.status === 200 ? object(object(result.value).properties).provisioningState : undefined;
  const phase: StorageDeployment["phase"] = state === "Succeeded" ? "ready" : state === "Failed" || state === "Canceled" ? "failed" : state === "Running" || state === "Accepted" ? "submitted" : "unknown";
  let networkAccess = d.networkAccess;
  if (phase === "ready") {
    const account = await control.request(record.input.subscriptionId, `${reportStorageNames(record).id}?api-version=2023-05-01`);
    const value = account.status === 200 ? object(object(account.value).properties).publicNetworkAccess : undefined;
    networkAccess = typeof value === "string" && ["Enabled", "Disabled", "SecuredByPerimeter"].includes(value) ? value : "Unknown";
  }
  const next = { ...record, storageDeployment: { ...d, phase, networkAccess } }; await control.persist(next); return next;
}
