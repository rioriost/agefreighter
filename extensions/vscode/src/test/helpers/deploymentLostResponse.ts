/** Test-companion only. No commands, credentials, network or production hooks. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, open, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";

export const lostResponseHash = (value: unknown): string => createHash("sha256").update(JSON.stringify(value)).digest("hex");
export interface LostResponsePermit {
  kind: "request" | "list"; path: string; method: "GET" | "POST"; maxCalls: number; bodySHA256?: string;
}
export interface LostResponsePlan {
  schemaVersion: 1; workflow: string; subscription: string; deploymentId: string;
  templateSHA256: string; previewSHA256: string; expiresAt: string;
  annotation: string; permits: LostResponsePermit[];
}
export async function retainLostResponseFile(root: string, name: string, value: unknown): Promise<void> {
  assert.match(name, /^[a-z0-9.-]+\.json$/);
  const file = await open(join(root, name), "wx", 0o600);
  try { await file.writeFile(JSON.stringify(value, null, 2) + "\n"); await file.sync(); } finally { await file.close(); }
  const directory = await open(root, "r");
  try { await directory.sync(); } finally { await directory.close(); }
}
export function validateLostResponsePlan(plan: LostResponsePlan, record: RunnerRecord): void {
  assert.equal(plan.schemaVersion, 1);
  assert.equal(plan.workflow, record.id); assert.equal(plan.subscription, record.input.subscriptionId);
  assert.equal(plan.deploymentId, record.deploymentId);
  assert.match(plan.workflow, /^[a-f0-9-]{36}$/);
  assert.match(plan.subscription, /^[a-f0-9-]{36}$/);
  assert.ok(plan.deploymentId.startsWith(`/subscriptions/${plan.subscription}/resourceGroups/`));
  assert.match(plan.deploymentId, /\/providers\/Microsoft\.Resources\/deployments\/af-[a-f0-9]{20}$/);
  assert.equal(plan.templateSHA256, lostResponseHash(record.template));
  assert.equal(plan.previewSHA256, lostResponseHash(record));
  assert.equal(record.phase, "previewed");
  assert.ok(Number.isFinite(Date.parse(plan.expiresAt)) && Date.parse(plan.expiresAt) <= Date.parse(record.expiresAt));
  assert.ok(plan.annotation.length >= 40 && plan.annotation.length <= 4000);
  const keys = new Set<string>();
  for (const permit of plan.permits) {
    assert.ok(permit.kind === "request" || permit.kind === "list");
    assert.ok(permit.method === "GET" || permit.method === "POST");
    assert.ok(permit.path.startsWith(`/subscriptions/${plan.subscription}/`) && !/[\r\n#]/.test(permit.path));
    assert.equal(new URL(permit.path, "https://management.azure.com").pathname + new URL(permit.path, "https://management.azure.com").search, permit.path);
    assert.ok(Number.isSafeInteger(permit.maxCalls) && permit.maxCalls > 0 && permit.maxCalls <= 100);
    if (permit.method === "POST") {
      assert.equal(permit.kind, "request");
      assert.equal(permit.path, `${plan.deploymentId}/whatIf?api-version=2022-09-01`);
      assert.match(permit.bodySHA256 ?? "", /^[a-f0-9]{64}$/);
    } else assert.equal(permit.bodySHA256, undefined);
    const key = `${permit.kind}:${permit.method}:${permit.path}`;
    assert.ok(!keys.has(key)); keys.add(key);
  }
}

/** Wrap an instance-bound real control in a separately reviewed companion.
 * The create-only PUT claim survives crashes and prevents replay by a new
 * adapter instance. A failed receipt write is uncertainty, never fault credit. */
export async function lostResponseControl(root: string, planInput: LostResponsePlan, record: RunnerRecord, actual: RunnerControl): Promise<RunnerControl> {
  assert.equal(await realpath(root), root);
  const info = await lstat(root); assert.ok(info.isDirectory() && !info.isSymbolicLink()); assert.equal(info.mode & 0o777, 0o700);
  validateLostResponsePlan(planInput, record);
  const plan = structuredClone(planInput), path = `${plan.deploymentId}?api-version=2022-09-01`;
  const bodySHA256 = lostResponseHash({ properties: { mode: "Incremental", template: record.template } });
  let persistedIntent = false;
  const fresh = () => assert.ok(Date.now() < Date.parse(plan.expiresAt), "Lost-response action window expired");
  const claimPermit = async (kind: "request" | "list", subscription: string, requestPath: string, method: string, body?: unknown) => {
    fresh(); assert.equal(subscription, plan.subscription);
    const permit = plan.permits.find(p => p.kind === kind && p.path === requestPath && p.method === method);
    assert.ok(permit, "Request is outside the reviewed lost-response stage");
    assert.equal(body === undefined ? undefined : lostResponseHash(body), permit.bodySHA256);
    const digest = lostResponseHash([kind, method, requestPath]);
    for (let index = 1; index <= permit.maxCalls; index++) {
      try {
        await retainLostResponseFile(root, `request-${digest}-${index}.json`, { kind, method, path: requestPath, at: new Date().toISOString() });
        return;
      } catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; }
    }
    throw Error("Reviewed request count exhausted; no automatic retry");
  };
  return {
    sleep: actual.sleep.bind(actual),
    persist: async next => {
      assert.equal(next.id, plan.workflow); assert.equal(next.deploymentId, plan.deploymentId);
      assert.equal(lostResponseHash(next.template), plan.templateSHA256);
      await actual.persist(next);
      if (next.phase === "deployment-submitted") persistedIntent = true;
    },
    list: async (subscription, requestPath) => { await claimPermit("list", subscription, requestPath, "GET"); return actual.list(subscription, requestPath); },
    request: async (subscription, requestPath, method = "GET", body) => {
      body = body === undefined ? undefined : structuredClone(body);
      if (method !== "PUT") {
        await claimPermit("request", subscription, requestPath, method, body);
        return actual.request(subscription, requestPath, method, body);
      }
      fresh(); assert.equal(subscription, plan.subscription); assert.equal(requestPath, path);
      assert.ok(persistedIntent, "Production deployment intent must be durable before PUT");
      assert.equal(lostResponseHash(body), bodySHA256);
      await retainLostResponseFile(root, "runner-put-intent.json", { workflow: plan.workflow, deploymentId: plan.deploymentId,
        method: "PUT", path, templateSHA256: plan.templateSHA256, at: new Date().toISOString() });
      const response = await actual.request(subscription, requestPath, method, body);
      if (![200, 201, 202].includes(response.status)) throw Error("No successful acceptance response; preserve evidence and stop");
      // Deliberately retain no raw payload, poll URL, headers, errors or token.
      await retainLostResponseFile(root, "azure-acceptance.json", { workflow: plan.workflow, deploymentId: plan.deploymentId,
        status: response.status, at: new Date().toISOString(), mechanism: "adapter-received-response-withheld-from-controller",
        independentARMObservationRequired: true, realServiceQualifiedByThisFile: false });
      throw Error("TEST COMPANION: accepted deployment response withheld from controller; reconcile the same ID without replay");
    }
  };
}

export async function retainedLostResponseAcceptance(root: string): Promise<unknown> {
  return JSON.parse(await readFile(join(root, "azure-acceptance.json"), "utf8"));
}
