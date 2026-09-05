import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { assertFreshPreview, bootstrapScript, object, parseRunnerInput, previewHash, releaseArtifact, RunnerInput, RunnerRecord, runnerNames, runnerTemplate, sourceLocations, sourceWorkflowDraft, validateWhatIf } from "../../core/runner";
import { deploymentResources, preflightRunner, refreshRunner, RunnerControl, submitRunner, whatIfRunner } from "../../core/runnerLifecycle";
import { runnerHTML } from "../../core/runnerView";
import { RunnerLockedError, RunnerStore } from "../../guided/runnerStore";

const sub = "11111111-1111-4111-8111-111111111111";
const id = "22222222-2222-4222-8222-222222222222";
const base = `/subscriptions/${sub}/resourceGroups/test`;
const input: RunnerInput = { subscriptionId: sub, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2",
  subnetId: `${base}/providers/Microsoft.Network/virtualNetworks/test/subnets/runner`, source: { type: "csv", location: "local" } };
const digest = "a".repeat(64);
const artifact = releaseArtifact("2.4.0", `${digest}  agefreighter_v2.4.0_linux_amd64.tar.gz`);
test("local source drafts require no release or ARM calls and cannot be deployed", async () => {
  const draft = sourceWorkflowDraft(id, input), f = fixture();
  assert.equal(draft.phase, "draft"); assert.equal(draft.artifact.sha256, "");
  assert.equal((await refreshRunner(f.control, draft)).phase, "draft");
  await assert.rejects(submitRunner(f.control, draft));
  assert.deepEqual(f.events, []);
});
function record(): RunnerRecord {
  const template = runnerTemplate(id, input, artifact, "ssh-ed25519 AAAA");
  return { schemaVersion: 2, id, phase: "previewed", input: structuredClone(input), artifact, ...runnerNames(id, input), template,
    hourlyComputeUSD: .1, previewHash: previewHash(template, input, .1), expiresAt: new Date(Date.now() + 600_000).toISOString(), updatedAt: new Date().toISOString() };
}
function fixture() {
  const events: string[] = [];
  const saved: RunnerRecord[] = [];
  let failPut = false;
  let state = "Succeeded";
  const control: RunnerControl = {
    sleep: async () => {},
    persist: async r => { events.push(`persist:${r.phase}`); saved.push(structuredClone(r)); },
    list: async (_sub, path) => path.includes('/skus?') ? [{ resourceType: "virtualMachines", name: input.size, family: "standardBSv2Family", locations: [input.region],
      capabilities: [{ name: "vCPUs", value: "2" }, { name: "MemoryGB", value: "8" }], locationInfo: [{ location: input.region, zones: ["1"] }], restrictions: [] }]
      : [{ name: { value: "cores" }, currentValue: 0, limit: 10 }, { name: { value: "standardBSv2Family" }, currentValue: 0, limit: 10 }],
    request: async (_sub, path, method = "GET") => {
      events.push(`${method}:${path}`);
      if (method === "PUT") { if (failPut) throw new Error("connection lost after acceptance"); return { status: 201, value: {} }; }
      if (path.includes('/whatIf?')) return { status: 200, value: { status: "Succeeded", changes: deploymentResources(record()).map(resourceId => ({ resourceId, changeType: "Create" })) } };
      if (path.startsWith(input.subnetId+'?')) return { status: 200, value: { properties: { delegations: [] } } };
      if (path.includes('/virtualNetworks/test?')) return { status: 200, value: { location: input.region } };
      if (path.startsWith(base+'?')) return { status: 200, value: {} };
      if (state !== "Succeeded" && path.includes('/deployments/')) return { status: 200, value: { properties: { provisioningState: state } } };
      return { status: 404, value: {} };
    }
  };
  return { control, events, saved, fail: () => { failPut = true; }, state: (s: string) => { state = s; } };
}

test("source selection enforces all four source/location combinations", () => {
  assert.deepEqual(sourceLocations("neo4j"), ["azure", "on-premises", "other-cloud"]);
  assert.deepEqual(sourceLocations("postgresql"), sourceLocations("neo4j"));
  assert.deepEqual(sourceLocations("cosmos-nosql"), ["azure"]);
  assert.deepEqual(sourceLocations("csv"), ["local"]);
  assert.deepEqual(parseRunnerInput(input), input);
  for (const source of [{ type: "cosmos-nosql", location: "local" }, { type: "csv", location: "azure" }, { type: "neo4j", location: "local" }, { type: "unknown", location: "local" }]) assert.throws(() => parseRunnerInput({ ...input, source }));
});
test("runner input rejects missing Azure identity, injection and unrelated subnet", () => {
  assert.throws(() => parseRunnerInput({ ...input, source: { type: "neo4j", location: "azure" } }));
  assert.throws(() => parseRunnerInput({ ...input, region: "eastus'; command" }));
  assert.throws(() => parseRunnerInput({ ...input, size: "Standard_E128_v5" }));
  assert.throws(() => parseRunnerInput({ ...input, subnetId: input.subnetId.replace(sub, id) }));
  assert.throws(() => parseRunnerInput({ ...input, source: { ...input.source, resourceId: `${base}/providers/x/y/z` } }));
});
test("only one matching released Linux artifact is accepted", () => {
  assert.equal(artifact.sha256, digest);
  assert.throws(() => releaseArtifact("2.3.0", ""));
  assert.throws(() => releaseArtifact("2.4.0", ""));
  assert.throws(() => releaseArtifact("2.4.0", `${digest}  agefreighter_v2.4.0_linux_amd64.tar.gz\n${digest}  agefreighter_v2.4.0_linux_amd64.tar.gz`));
  assert.throws(() => bootstrapScript({ ...artifact, url: "https://example.com/exec" }));
  const script = bootstrapScript(artifact);
  assert.ok(script.indexOf('sha256sum --check') < script.indexOf('tar -xOzf'));
  assert.match(script, /inventory --help/);
  assert.doesNotMatch(script, /git clone|latest|curl.*\|.*sh|rm -/);
});
test("template uses private persistent Linux runner and no source credentials", () => {
  const template = record().template;
  const text = JSON.stringify(template);
  assert.doesNotMatch(text, /publicIPAddresses|adminPassword|roleAssignments|ephemeralOSDisk|flexibleServers/);
  const resources = template.resources as Record<string, unknown>[];
  assert.equal(resources.length, 3);
  const vm = resources.find(r => r.type === "Microsoft.Compute/virtualMachines")!;
  assert.deepEqual(vm.zones, ["1"]);
  const properties = object(vm.properties);
  assert.equal(object(properties.hardwareProfile).vmSize, "Standard_B2s_v2");
  assert.equal(object(object(properties.storageProfile).osDisk).deleteOption, "Detach");
  const config = Buffer.from(String(object(properties.osProfile).customData), "base64").toString();
  assert.match(config, /^#cloud-config/);
  assert.doesNotMatch(config, /password|token|neo4j:|postgres:\/\//i);
});
test("stale, modified, redirected and replayed previews fail closed", () => {
  assertFreshPreview(record());
  assert.throws(() => assertFreshPreview({ ...record(), expiresAt: "2000-01-01T00:00:00Z" }));
  assert.throws(() => assertFreshPreview({ ...record(), phase: "deployment-submitted" }));
  assert.throws(() => assertFreshPreview({ ...record(), hourlyComputeUSD: .01 }));
  assert.throws(() => assertFreshPreview({ ...record(), vmId: "different" }));
  assert.throws(() => assertFreshPreview({ ...record(), template: {} }));
});
test("what-if accepts exactly the reviewed creates, never modifies or deletes", () => {
  const ids = deploymentResources(record());
  const changes = ids.map(resourceId => ({ resourceId, changeType: "Create" }));
  validateWhatIf({ status: "Succeeded", changes }, ids);
  for (const changeType of ["Modify", "Delete", "NoChange", "Ignore"]) assert.throws(() => validateWhatIf({ status: "Succeeded", changes: changes.map(c => ({ ...c, changeType })) }, ids));
  assert.throws(() => validateWhatIf({ status: "Succeeded", changes: [] }, ids));
  assert.throws(() => validateWhatIf({ status: "Succeeded", changes: [...changes, changes[0]] }, ids));
  assert.throws(() => validateWhatIf({ status: "Running", changes }, ids));
});
test("discovery preflight checks region, zone, delegation and both quotas", async () => {
  await preflightRunner(fixture().control, input);
  await assert.rejects(preflightRunner(fixture().control, { ...input, zone: "3" }), /zone/);
  const f = fixture(); f.control.list = async () => [];
  await assert.rejects(preflightRunner(f.control, input), /not available/);
  const g = fixture(); const request = g.control.request;
  g.control.request = async (...args) => args[1].startsWith(input.subnetId+'?') ? { status: 200, value: { properties: { delegations: [{}] } } } : request(...args);
  await assert.rejects(preflightRunner(g.control, input), /non-delegated/);
  const h = fixture(); const list = h.control.list;
  h.control.list = async (...args) => args[1].includes('/usages?') ? [] : list(...args);
  await assert.rejects(preflightRunner(h.control, input), /quota/);
});
test("existing resource collision prevents what-if and deployment", async () => {
  const f = fixture(); const request = f.control.request;
  f.control.request = async (...args) => args[1].includes('/networkInterfaces/') ? { status: 200, value: {} } : request(...args);
  await assert.rejects(whatIfRunner(f.control, record()), /already exists/);
  assert.ok(!f.events.some(e => e.startsWith('POST:') || e.startsWith('PUT:')));
});

test("source placement uses PostgreSQL availabilityZone and rejects cross-subscription guesses", async () => {
  const f = fixture(); const request = f.control.request;
  const sourceId = `${base}/providers/Microsoft.DBforPostgreSQL/flexibleServers/source`;
  const source = { type: "postgresql" as const, location: "azure" as const, resourceId: sourceId };
  f.control.request = async (...args) => args[1].startsWith(sourceId+'?') ? { status: 200, value: { location: input.region, properties: { availabilityZone: "2" } } } : request(...args);
  await assert.rejects(preflightRunner(f.control, { ...input, source }), /source availability zone/);
  await assert.rejects(preflightRunner(f.control, { ...input, source: { ...source, resourceId: sourceId.replace(sub,id) } }), /physical-zone mapping/);
});

test("Cosmos location is the selected data region, not account metadata", async () => {
  const f = fixture(); const request = f.control.request;
  const sourceId = `${base}/providers/Microsoft.DocumentDB/databaseAccounts/source`;
  const source = { type: "cosmos-nosql" as const, location: "azure" as const, resourceId: sourceId };
  f.control.request = async (...args) => args[1].startsWith(sourceId+'?') ? { status: 200, value: { location: "westus", properties: { locations: [{ locationName: "Japan East" }] } } } : request(...args);
  await preflightRunner(f.control, { ...input, source });
  f.control.request = async (...args) => args[1].startsWith(sourceId+'?') ? { status: 200, value: { location: input.region, properties: { locations: [{ locationName: "West US" }] } } } : request(...args);
  await assert.rejects(preflightRunner(f.control, { ...input, source }), /actual Cosmos data region/);
});

test("pending what-if is polled without re-POSTing and incomplete evidence blocks", async () => {
  const f = fixture(); const request = f.control.request;
  const poll = `/subscriptions/${sub}/providers/Microsoft.Resources/locations/japaneast/operationresults/test?api-version=2022-09-01`;
  let posts = 0;
  f.control.request = async (...args) => {
    if (args[2] === 'POST') { posts++; return { status: 202, value: {}, poll }; }
    if (args[1] === poll) return { status: 200, value: { status: "Succeeded", changes: deploymentResources(record()).map(resourceId => ({ resourceId, changeType: "Create" })) } };
    return request(...args);
  };
  await whatIfRunner(f.control, record()); assert.equal(posts, 1);
  f.control.request = async (...args) => args[2] === 'POST' ? { status: 200, value: { status: "Succeeded" } } : request(...args);
  await assert.rejects(whatIfRunner(f.control, record()), /complete change review/);
});
test("submission records durable intent before its single Azure PUT", async () => {
  const f = fixture(); const result = await submitRunner(f.control, record());
  assert.equal(result.phase, "deployment-submitted");
  assert.ok(f.events.findIndex(e => e === 'persist:deployment-submitted') < f.events.findIndex(e => e.startsWith('PUT:')));
  assert.equal(f.events.filter(e => e.startsWith('PUT:')).length, 1);
  await assert.rejects(submitRunner(f.control, result), /stale|submitted/);
  assert.equal(f.events.filter(e => e.startsWith('PUT:')).length, 1);
});
test("ambiguous acceptance is retained and refresh never replays PUT", async () => {
  const f = fixture(); f.fail();
  const result = await submitRunner(f.control, record());
  assert.equal(result.phase, "unknown"); assert.equal(f.saved.at(-1)?.phase, "unknown");
  await assert.rejects(submitRunner(f.control, result));
  const observed = await refreshRunner(f.control, result);
  assert.equal(observed.phase, "unknown");
  assert.equal(f.events.filter(e => e.startsWith('PUT:')).length, 1);
  f.state("Failed"); assert.equal((await refreshRunner(f.control, result)).phase, "failed");
});
test("persistence failure prevents deployment", async () => {
  const f = fixture(); f.control.persist = async () => { throw new Error("disk unavailable"); };
  await assert.rejects(submitRunner(f.control, record()), /disk unavailable/);
  assert.ok(!f.events.some(e => e.startsWith('PUT:')));
});
test("control-plane success does not claim guest assessment completion", async () => {
  const f = fixture(); f.control.request = async () => ({ status: 200, value: { properties: { provisioningState: "Succeeded" } } });
  const result = await refreshRunner(f.control, { ...record(), phase: "deployment-submitted" });
  assert.equal(result.phase, "provisioned");
  assert.ok(!('jobId' in result));
});

test("runner webview script parses and provides no local password or load action", () => {
  const html = runnerHTML("https://webview.example");
  const code = /<script nonce="[^"]+">([\s\S]+)<\/script>/.exec(html)?.[1];
  assert.ok(code); new Script(code);
  assert.match(html, /default-src 'none'/);
  assert.doesNotMatch(html, /unsafe-inline|onchange=|onclick=|type="password"|Connect and profile source/);
  for (const type of ["neo4j", "postgresql", "cosmos-nosql", "csv"]) assert.ok(html.includes(`value="${type}"`));
  assert.match(html, /No desktop AGEFreighter installation/);
  assert.match(html, /Configure source & assessment/);
  assert.match(html, /Check Linux guest readiness/);
});

test("atomic runner records preserve distinct workflows and exclude group write permissions", async () => {
  const root = await mkdtemp(join(tmpdir(), "agefreighter-runner-test-"));
  try {
    const store = new RunnerStore(root); const first = record(); const second = { ...record(), id: "33333333-3333-4333-8333-333333333333" };
    await Promise.all([store.write(first), store.write(second)]);
    assert.equal((await store.list()).length, 2);
    assert.deepEqual(await store.read(first.id), first);
    assert.equal((await stat(join(root, first.id + '.json'))).mode & 0o077, 0);
    await assert.rejects(store.read('../outside'), /Invalid/);
    assert.equal(JSON.parse(await readFile(join(root, first.id + '.json'), "utf8")).phase, 'previewed');
  } finally { await rm(root, { recursive: true }); }
});

test("cross-window runner lock excludes simultaneous submit and releases after errors", async () => {
  const root = await mkdtemp(join(tmpdir(), "agefreighter-runner-test-"));
  try {
    const first = new RunnerStore(root); const second = new RunnerStore(root);
    await first.exclusive(id, async () => {
      await assert.rejects(second.exclusive(id, async () => {}), RunnerLockedError);
    });
    await assert.rejects(first.exclusive(id, async () => { throw new Error('preflight failed'); }), /preflight failed/);
    assert.equal(await second.exclusive(id, async () => 'safe'), 'safe');
  } finally { await rm(root, { recursive: true }); }
});
