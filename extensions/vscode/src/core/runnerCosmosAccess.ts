import { randomUUID } from "node:crypto";
import { object } from "./runner";
import type { RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";

const readerRole = "00000000-0000-0000-0000-000000000001";
const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;

export interface CosmosAccess {
  phase: "previewed" | "submitted" | "unknown" | "ready";
  assignmentId: string;
  roleDefinitionId: string;
  scope: string;
  principalId: string;
  submittedAt?: string;
}

function account(record: RunnerRecord): string {
  const id = record.input.source.resourceId;
  if (record.input.source.type !== "cosmos-nosql" || record.input.source.location !== "azure" || !id || !/\/providers\/Microsoft\.DocumentDB\/databaseAccounts\/[^/]+$/i.test(id)) throw new Error("Select one Azure Cosmos DB account for this workflow.");
  return id;
}

function valid(access: CosmosAccess, record: RunnerRecord): boolean {
  const scope = account(record), prefix = `${scope}/sqlRoleAssignments/`;
  return uuid.test(access.principalId) && access.scope.toLowerCase() === scope.toLowerCase() &&
    access.roleDefinitionId.toLowerCase() === `${scope}/sqlRoleDefinitions/${readerRole}`.toLowerCase() &&
    access.assignmentId.toLowerCase().startsWith(prefix.toLowerCase()) && uuid.test(access.assignmentId.slice(prefix.length));
}

function matches(value: unknown, access: CosmosAccess): boolean {
  const properties = object(object(value).properties);
  return String(properties.principalId).toLowerCase() === access.principalId.toLowerCase() &&
    String(properties.roleDefinitionId).toLowerCase() === access.roleDefinitionId.toLowerCase() &&
    String(properties.scope).toLowerCase() === access.scope.toLowerCase();
}

export async function previewCosmosAccess(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  if (record.phase !== "provisioned" || record.cosmosAccess) throw new Error("Cosmos access can be previewed once after runner provisioning.");
  const response = await control.request(record.input.subscriptionId, `${record.vmId}?api-version=2024-07-01`), vm = object(response.value), identity = object(vm.identity), tags = object(vm.tags);
  if (response.status !== 200 || tags.workflow !== record.id || tags.application !== "agefreighter" || identity.type !== "SystemAssigned" || typeof identity.principalId !== "string" || !uuid.test(identity.principalId)) throw new Error("The owned runner managed identity is unavailable.");
  const scope = account(record), assignmentId = `${scope}/sqlRoleAssignments/${randomUUID()}`;
  if ((await control.request(record.input.subscriptionId, `${assignmentId}?api-version=2024-05-15`)).status !== 404) throw new Error("The proposed Cosmos role assignment already exists.");
  const cosmosAccess: CosmosAccess = { phase: "previewed", assignmentId, scope, principalId: identity.principalId, roleDefinitionId: `${scope}/sqlRoleDefinitions/${readerRole}` };
  const next = { ...record, cosmosAccess };
  await control.persist(next); return next;
}

export async function submitCosmosAccess(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const access = record.cosmosAccess;
  if (!access || access.phase !== "previewed" || !valid(access, record)) throw new Error("Review the retained Cosmos read-only grant first.");
  if ((await control.request(record.input.subscriptionId, `${access.assignmentId}?api-version=2024-05-15`)).status !== 404) throw new Error("Cosmos role assignment collision; no write was attempted.");
  const next: RunnerRecord = { ...record, cosmosAccess: { ...access, phase: "submitted", submittedAt: new Date().toISOString() } };
  await control.persist(next);
  try {
    const response = await control.request(record.input.subscriptionId, `${access.assignmentId}?api-version=2024-05-15`, "PUT", { properties: { principalId: access.principalId, roleDefinitionId: access.roleDefinitionId, scope: access.scope } });
    if (response.status < 200 || response.status >= 300) throw new Error();
  } catch {
    next.cosmosAccess!.phase = "unknown"; await control.persist(next);
  }
  return next;
}

export async function refreshCosmosAccess(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const access = record.cosmosAccess;
  if (!access || !valid(access, record)) throw new Error("No valid retained Cosmos access request exists.");
  const response = await control.request(record.input.subscriptionId, `${access.assignmentId}?api-version=2024-05-15`);
  if (response.status === 404) {
    if (access.phase === "ready") throw new Error("The verified Cosmos Data Reader assignment no longer exists.");
    return record;
  }
  if (response.status !== 200 || !matches(response.value, access)) throw new Error("Cosmos data-plane role assignment identity changed.");
  const next: RunnerRecord = { ...record, cosmosAccess: { ...access, phase: "ready" } };
  await control.persist(next); return next;
}

export async function assertCosmosAccessCurrent(control: RunnerControl, record: RunnerRecord): Promise<void> {
  if (record.input.source.type !== "cosmos-nosql") return;
  const access = record.cosmosAccess;
  if (!access || access.phase !== "ready" || !valid(access, record)) throw new Error("The retained Cosmos Data Reader grant is not ready.");
  const response = await control.request(record.input.subscriptionId, `${access.assignmentId}?api-version=2024-05-15`);
  if (response.status !== 200 || !matches(response.value, access)) throw new Error("The Cosmos Data Reader assignment is missing or changed; no source operation was submitted.");
}

export function cosmosAccessReady(record: RunnerRecord): boolean {
  return record.input.source.type !== "cosmos-nosql" || !!record.cosmosAccess && record.cosmosAccess.phase === "ready" && valid(record.cosmosAccess, record);
}
