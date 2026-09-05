import { assertFreshPreview, object, RunnerInput, RunnerRecord, runnerNames, validateWhatIf } from "./runner";
import { parseComputeSkus, parseQuotaUsages } from "./proposal";

export interface RunnerControl {
  request(subscription: string, path: string, method?: "GET" | "POST" | "PUT", body?: unknown): Promise<{ status: number; value: unknown; poll?: string }>;
  list(subscription: string, path: string): Promise<unknown[]>;
  persist(record: RunnerRecord): Promise<void>;
  sleep(ms: number): Promise<void>;
}

export function deploymentResources(record: RunnerRecord): string[] {
  const { prefix } = runnerNames(record.id, record.input);
  const base = `/subscriptions/${record.input.subscriptionId}/resourceGroups/${record.input.resourceGroup}/providers`;
  return [`${base}/Microsoft.Network/networkSecurityGroups/${prefix}`, `${base}/Microsoft.Network/networkInterfaces/${prefix}`, record.vmId];
}

export async function preflightRunner(control: RunnerControl, input: RunnerInput): Promise<void> {
  const sub = input.subscriptionId;
  const base = `/subscriptions/${sub}`;
  const subnet = await control.request(sub, `${input.subnetId}?api-version=2024-05-01`);
  if (subnet.status === 404) throw new Error("The selected subnet does not exist.");
  const subnetProperties = object(object(subnet.value).properties);
  if (!Array.isArray(subnetProperties.delegations) || subnetProperties.delegations.length) throw new Error("Use a non-delegated compute subnet, not the Flexible Server delegated subnet.");
  const vnet = await control.request(sub, `${input.subnetId.replace(/\/subnets\/[^/]+$/i, '')}?api-version=2024-05-01`);
  if (vnet.status === 404 || object(vnet.value).location !== input.region) throw new Error("The runner region must match the existing VNet.");
  const group = await control.request(sub, `${base}/resourceGroups/${input.resourceGroup}?api-version=2021-04-01`);
  if (group.status === 404) throw new Error("Select an existing resource group.");
  const skus = parseComputeSkus({ value: await control.list(sub, `${base}/providers/Microsoft.Compute/skus?api-version=2021-07-01&$filter=${encodeURIComponent(`location eq '${input.region}'`)}`) }, input.region);
  const sku = skus.find(s => s.name === input.size);
  if (!sku || sku.restricted || !sku.zones.includes(input.zone)) throw new Error("The selected discovery SKU is not available in this subscription/zone.");
  const quota = parseQuotaUsages({ value: await control.list(sub, `${base}/providers/Microsoft.Compute/locations/${input.region}/usages?api-version=2025-04-01`) });
  for (const name of ["cores", sku.family]) {
    const usage = quota.find(q => q.name.toLowerCase() === name.toLowerCase());
    if (!usage || usage.limit - usage.current < sku.vCores) throw new Error("Regional or VM-family quota is insufficient or unavailable.");
  }
  if (input.source.location === "azure") {
    const id = input.source.resourceId!;
    // Logical zone numbers are subscription scoped. Cross-subscription placement
    // needs an explicit physical-zone mapping, not a guessed matching number.
    if (id.split("/")[2]?.toLowerCase() !== sub.toLowerCase()) throw new Error("Cross-subscription source placement requires physical-zone mapping and is not supported yet.");
    const path = id.toLowerCase();
    const vm = /\/providers\/microsoft.compute\/virtualmachines\/[^/]+$/.test(path);
    const pg = /\/providers\/microsoft.dbforpostgresql\/flexibleservers\/[^/]+$/.test(path);
    const cosmos = /\/providers\/microsoft.documentdb\/databaseaccounts\/[^/]+$/.test(path);
    if (!(input.source.type === "neo4j" && vm || input.source.type === "postgresql" && (vm || pg) || input.source.type === "cosmos-nosql" && cosmos)) throw new Error("The selected Azure resource is not a candidate for this source type.");
    const source = await control.request(sub, `${id}?api-version=${vm ? "2024-07-01" : pg ? "2024-08-01" : "2024-05-15"}`);
    if (source.status === 404) throw new Error("The source resource was not found.");
    const value = object(source.value);
    // Cosmos account resource location is metadata; use its actual read/write regions.
    if (cosmos) {
      const properties = object(value.properties);
      const regions = Array.isArray(properties.locations) ? properties.locations.map(x => String(object(x).locationName).replaceAll(" ", "").toLowerCase()) : [];
      if (!regions.includes(input.region)) throw new Error("Select an actual Cosmos data region. Account metadata location is not placement evidence.");
    } else {
      if (value.location !== input.region) throw new Error("Select the source region for the runner.");
      const properties = object(value.properties);
      const zone = pg ? properties.availabilityZone : Array.isArray(value.zones) && value.zones.length === 1 ? value.zones[0] : undefined;
      if (zone && zone !== input.zone) throw new Error("Select the source availability zone for the runner.");
    }
  }
}

export async function whatIfRunner(control: RunnerControl, record: RunnerRecord): Promise<void> {
  const sub = record.input.subscriptionId;
  const ids = deploymentResources(record);
  for (const id of [...ids, `${record.deploymentId.slice(0, record.deploymentId.indexOf('/providers/'))}/providers/Microsoft.Compute/disks/${runnerNames(record.id, record.input).prefix}-os`, record.deploymentId]) {
    const version = id.includes("/Microsoft.Network/") ? "2024-05-01" : id.includes("/deployments/") ? "2022-09-01" : id.includes("/disks/") ? "2024-03-02" : "2024-07-01";
    if ((await control.request(sub, `${id}?api-version=${version}`)).status !== 404) throw new Error("A proposed resource already exists. No existing resources will be overwritten.");
  }
  let response = await control.request(sub, `${record.deploymentId}/whatIf?api-version=2022-09-01`, "POST", {
    properties: { mode: "Incremental", template: record.template, whatIfSettings: { resultFormat: "ResourceIdOnly" } }
  });
  for (let attempt = 0; attempt < 30; attempt++) {
    const value = object(response.value);
    if (value.status === "Succeeded" && Array.isArray(value.changes)) { validateWhatIf(value, ids); return; }
    if (value.status === "Failed" || value.status === "Canceled" || !response.poll) throw new Error("Azure what-if did not produce a complete change review.");
    await control.sleep(2000);
    const poll = response.poll;
    response = await control.request(sub, poll);
    response.poll ??= poll;
  }
  throw new Error("Azure what-if is still pending. No deployment was submitted.");
}

/** Caller holds a single-flight lock, obtains explicit approval, then calls this. */
export async function submitRunner(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  assertFreshPreview(record);
  await preflightRunner(control, record.input);
  await whatIfRunner(control, record);
  assertFreshPreview(record);
  const submitted: RunnerRecord = { ...record, phase: "deployment-submitted", updatedAt: new Date().toISOString() };
  // Persist the intent BEFORE the mutating call; ambiguous failure must never replay it.
  await control.persist(submitted);
  try {
    const response = await control.request(record.input.subscriptionId, `${record.deploymentId}?api-version=2022-09-01`, "PUT", { properties: { mode: "Incremental", template: record.template } });
    if (response.status === 404) throw new Error("Deployment was not accepted.");
    return submitted;
  } catch {
    const unknown: RunnerRecord = { ...submitted, phase: "unknown" };
    await control.persist(unknown);
    return unknown;
  }
}

export async function refreshRunner(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  if (record.phase === "previewed" || record.phase === "draft") return record;
  const result = await control.request(record.input.subscriptionId, `${record.deploymentId}?api-version=2022-09-01`);
  const state = result.status === 404 ? undefined : object(object(result.value).properties).provisioningState;
  const phase = state === "Succeeded" ? "provisioned" : state === "Failed" || state === "Canceled" ? "failed" : state === "Running" || state === "Accepted" ? "deployment-submitted" : "unknown";
  const next: RunnerRecord = { ...record, phase, updatedAt: new Date().toISOString() };
  await control.persist(next);
  return next;
}
