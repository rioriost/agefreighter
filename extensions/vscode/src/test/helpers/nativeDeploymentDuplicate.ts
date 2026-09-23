/** Pure fixture/evidence helpers. Inert responses are not Azure observations. */
import assert from "node:assert/strict";
import { RunnerRecord, runnerNames } from "../../core/runner";
import { deploymentResources } from "../../core/runnerLifecycle";

export type DuplicateMode = "same-window" | "two-window";
export function duplicateReadFixture(record: RunnerRecord) {
  const { input } = record, base = `/subscriptions/${input.subscriptionId}`, group = `${base}/resourceGroups/${input.resourceGroup}`;
  const requests = new Map<string, { status: number; value: unknown }>([
    [`${input.subnetId}?api-version=2024-05-01`, { status: 200, value: { properties: { delegations: [] } } }],
    [`${input.subnetId.replace(/\/subnets\/[^/]+$/i, "")}?api-version=2024-05-01`, { status: 200, value: { location: input.region } }],
    [`${group}?api-version=2021-04-01`, { status: 200, value: {} }]
  ]);
  for (const id of [...deploymentResources(record), `${group}/providers/Microsoft.Compute/disks/${runnerNames(record.id, input).prefix}-os`, record.deploymentId]) {
    const version = id.includes("/Microsoft.Network/") ? "2024-05-01" : id.includes("/deployments/") ? "2022-09-01" : id.includes("/disks/") ? "2024-03-02" : "2024-07-01";
    requests.set(`${id}?api-version=${version}`, { status: 404, value: {} });
  }
  const family = "standardBSv2Family";
  const lists = new Map<string, unknown[]>([
    [`${base}/providers/Microsoft.Compute/skus?api-version=2021-07-01&$filter=${encodeURIComponent(`location eq '${input.region}'`)}`, [{ resourceType: "virtualMachines", name: input.size, family,
      locations: [input.region], capabilities: [{ name: "vCPUs", value: "2" }, { name: "MemoryGB", value: "8" }], locationInfo: [{ location: input.region, zones: [input.zone] }], restrictions: [] }]],
    [`${base}/providers/Microsoft.Compute/locations/${input.region}/usages?api-version=2025-04-01`, ["cores", family].map(value => ({ name: { value }, currentValue: 0, limit: 10 }))],
    [`${group}/resources?api-version=2021-04-01`, []]
  ]);
  return { requests, lists, whatIf: { status: "Succeeded", properties: { changes: deploymentResources(record).map(resourceId => ({ resourceId, changeType: "Create" })) } } };
}

export function assertOnlySubmittedIntent(original: RunnerRecord, next: RunnerRecord): void {
  assert.equal(next.phase, "deployment-submitted");
  assert.ok(Number.isFinite(Date.parse(next.updatedAt)) && Date.parse(next.updatedAt) >= Date.parse(original.updatedAt));
  assert.deepEqual({ ...next, phase: original.phase, updatedAt: original.updatedAt }, original);
}

export interface DuplicateObservation {
  role: "A" | "B"; mode: DuplicateMode; hostPID: number; sessionId: string; profile: string; storeRoot: string;
  deployEntries: number; modalEntries: number; submitEntries: number; storeWrites: number; inertPUTs: number;
  modalOpenedAt?: string; modalReturnedAt?: string; nativeChoice?: string;
  errors: string[]; denied: string[]; initialSHA256: string; modalSHA256?: string; finalSHA256?: string;
  expiresAt: string;
}
export function duplicateObservationFailures(a: DuplicateObservation, b?: DuplicateObservation): string[] {
  const failures: string[] = [];
  for (const host of b ? [a, b] : [a]) {
    const opened = Date.parse(host.modalOpenedAt ?? ""), returned = Date.parse(host.modalReturnedAt ?? ""), expires = Date.parse(host.expiresAt);
    if (![opened, returned, expires].every(Number.isFinite) || opened > returned || returned >= expires) failures.push("Duplicate evidence must complete before original expiry, not qualify through the expiry guard.");
  }
  if (a.role !== "A" || a.deployEntries < 1 || a.modalEntries !== 1 || a.submitEntries !== 1 || a.storeWrites !== 1 || a.inertPUTs !== 1 || a.nativeChoice !== "Create reviewed runner" || a.errors.length || a.denied.length) failures.push("Primary host did not retain exactly one approved inert submission.");
  if (!a.finalSHA256 || a.finalSHA256 === a.initialSHA256 || a.modalSHA256 !== a.initialSHA256) failures.push("Primary original review or legitimate intent store transition is missing.");
  if (a.mode === "two-window") {
    if (!b || b.role !== "B" || b.hostPID === a.hostPID || b.profile !== a.profile || b.storeRoot !== a.storeRoot || !a.sessionId || !b.sessionId) failures.push("Two distinct observed hosts in the same profile and physical store are required.");
    if (b && (b.expiresAt !== a.expiresAt || b.initialSHA256 !== a.initialSHA256 || b.modalSHA256 !== a.initialSHA256 || b.finalSHA256 !== a.finalSHA256 ||
      !(Date.parse(b.modalOpenedAt ?? "") <= Date.parse(a.modalReturnedAt ?? "")) || !(Date.parse(b.modalReturnedAt ?? "") >= Date.parse(a.modalReturnedAt ?? "")))) failures.push("Secondary original approval was not retained across the primary intent transition.");
    if (b && (b.deployEntries < 1 || b.modalEntries !== 1 || b.submitEntries || b.storeWrites || b.inertPUTs || b.nativeChoice !== "Create reviewed runner" || b.denied.length || b.errors.length !== 1 || !b.errors[0]?.includes("approved preview changed or expired"))) failures.push("Secondary approval did not refuse before all submission boundaries.");
  }
  return failures;
}
