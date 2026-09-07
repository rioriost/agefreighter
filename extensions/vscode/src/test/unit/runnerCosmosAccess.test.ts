import assert from "node:assert/strict";
import test from "node:test";
import { sourceWorkflowDraft } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { assertCosmosAccessCurrent, cosmosAccessReady, previewCosmosAccess, refreshCosmosAccess, submitCosmosAccess } from "../../core/runnerCosmosAccess";

const id = "11111111-1111-4111-8111-111111111111", principal = "22222222-2222-4222-8222-222222222222";
const account = `/subscriptions/${id}/resourceGroups/source/providers/Microsoft.DocumentDB/databaseAccounts/p1source`;

function fixture() {
  const record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "migration", region: "japaneast", zone: "1", subnetId: "subnet", size: "Standard_B2s_v2", source: { type: "cosmos-nosql", location: "azure", resourceId: account } });
  record.phase = "provisioned";
  const requests: { path: string; method: string; body?: unknown }[] = [], saved: unknown[] = [];
  let assignment: unknown, lose = false;
  const control: RunnerControl = { sleep: async () => {}, list: async () => [], persist: async value => { saved.push(structuredClone(value)); }, request: async (_subscription, path, method = "GET", body) => {
    requests.push({ path, method, body });
    if (path === `${record.vmId}?api-version=2024-07-01`) return { status: 200, value: { identity: { type: "SystemAssigned", principalId: principal }, tags: { workflow: id, application: "agefreighter" } } };
    if (method === "PUT") { assignment = body; if (lose) throw new Error("lost response"); return { status: 201, value: {} }; }
    if (assignment) return { status: 200, value: assignment };
    return { status: 404, value: {} };
  } };
  return { record, control, requests, saved, lose: () => { lose = true; } };
}

test("Cosmos Data Reader grant is account-scoped, persist-first and GET-verified", async () => {
  const f = fixture(), preview = await previewCosmosAccess(f.control, f.record);
  assert.equal(preview.cosmosAccess?.phase, "previewed"); assert.equal(cosmosAccessReady(preview), false);
  const submitted = await submitCosmosAccess(f.control, preview);
  assert.equal(submitted.cosmosAccess?.phase, "submitted");
  const put = f.requests.find(request => request.method === "PUT")!;
  assert.equal((put.body as any).properties.scope, account); assert.match((put.body as any).properties.roleDefinitionId, /00000000-0000-0000-0000-000000000001$/);
  assert.equal((f.saved.at(-1) as any).cosmosAccess.phase, "submitted");
  const ready = await refreshCosmosAccess(f.control, submitted);
  assert.equal(ready.cosmosAccess?.phase, "ready"); assert.equal(cosmosAccessReady(ready), true);
  await assertCosmosAccessCurrent(f.control, ready);
});

test("a removed or changed ready assignment blocks later source operations", async () => {
  const f = fixture(), preview = await previewCosmosAccess(f.control, f.record), submitted = await submitCosmosAccess(f.control, preview), ready = await refreshCosmosAccess(f.control, submitted);
  ready.cosmosAccess!.principalId = "33333333-3333-4333-8333-333333333333";
  await assert.rejects(() => assertCosmosAccessCurrent(f.control, ready), /missing or changed/);
});

test("lost Cosmos role PUT is retained and reconciled without replay", async () => {
  const f = fixture(), preview = await previewCosmosAccess(f.control, f.record); f.lose();
  const unknown = await submitCosmosAccess(f.control, preview); assert.equal(unknown.cosmosAccess?.phase, "unknown");
  const writes = f.requests.filter(request => request.method === "PUT").length;
  const ready = await refreshCosmosAccess(f.control, unknown);
  assert.equal(ready.cosmosAccess?.phase, "ready"); assert.equal(f.requests.filter(request => request.method === "PUT").length, writes);
});
