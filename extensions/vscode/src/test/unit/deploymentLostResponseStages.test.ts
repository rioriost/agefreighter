import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, { TestContext } from "node:test";
import { RunnerRecord, previewHash, runnerTemplate, sourceWorkflowDraft } from "../../core/runner";
import { developmentArtifact } from "../../core/runnerDevelopment";
import { dispatchGuest, guestReadinessScript, reconcileGuest } from "../../core/runnerGuest";
import { deploymentResources, refreshRunner, RunnerControl, submitRunner } from "../../core/runnerLifecycle";
import { reportStorageNames } from "../../core/runnerReportStorage";
import { refreshStorage, storageDraft, submitStorage } from "../../core/runnerStorageLifecycle";
import type { AzureSession } from "../../guided/azure";
import { LostResponseScope, LostResponseStages } from "../helpers/deploymentLostResponseStages";

const id = "11111111-1111-4111-8111-111111111111", principal = "22222222-2222-4222-8222-222222222222";
const operation = "33333333-3333-4333-8333-333333333333";
const group = `/subscriptions/${id}/resourceGroups/test`, commit = "a".repeat(40), sha = "b".repeat(64);

async function fixture(t: TestContext) {
  const root = await mkdtemp(join(tmpdir(), "af-lost-stages-"));
  t.after(() => rm(root, {recursive: true, force: true}));
  let record = sourceWorkflowDraft(id, {subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1",
    subnetId: `${group}/providers/Microsoft.Network/virtualNetworks/test/subnets/runner`, size: "Standard_B2s_v2", source: {type: "csv", location: "local"}});
  const names = reportStorageNames(record);
  const artifact = developmentArtifact(record, {schemaVersion: 1, platform: "linux-amd64", version: `2.4.0-dev.${commit.slice(0, 12)}`, commit, sha256: sha, bytes: 42});
  const scope: LostResponseScope = {workflow: id, input: structuredClone(record.input), artifactSHA256: sha, artifactBytes: 42,
    manifestPath: join(root, "manifest.json"), archivePath: join(root, "negative.tar.gz"), annotation: "Intentional unusable B09 packaging fixture. New disposable VM only.", expiresAt: new Date(Date.now() + 900000).toISOString()};
  const events: {method: string; path?: string; body?: unknown}[] = [], saved: RunnerRecord[] = [];
  let storageAccepted = false, runnerAccepted = false, readinessAccepted = false, acceptanceStatus = 201;
  const actual = {
    runnerRequest: async (_subscription: string, path: string, method = "GET", body?: unknown) => {
      events.push({method, path, body: structuredClone(body)});
      if (method === "PUT") {
        if (path.includes("-transfer?")) storageAccepted = true;
        else if (path.includes("/runCommands/")) readinessAccepted = true;
        else runnerAccepted = true;
        return {status: acceptanceStatus, value: {privateDiagnostic: "must not be retained"}};
      }
      if (method === "POST") {
        const resources = path.includes("-transfer/whatIf") ? [names.id, names.containerId, record.storageDeployment!.roleId] : deploymentResources(record);
        return {status: 200, value: {status: "Succeeded", properties: {changes: resources.map(resourceId => ({resourceId, changeType: "Create"}))}}};
      }
      if (path.startsWith(record.input.subnetId + "?")) return {status: 200, value: {properties: {delegations: []}}};
      if (path.startsWith(record.input.subnetId.replace(/\/subnets\/[^/]+$/, "") + "?")) return {status: 200, value: {location: "japaneast"}};
      if (path.startsWith(group + "?")) return {status: 200, value: {}};
      if (storageAccepted && path.startsWith(names.id + "?")) return {status: 200, value: {id: names.id, location: "japaneast", tags: {application: "agefreighter", workflow: id, purpose: "artifact-transfer"}, properties: {
        provisioningState: "Succeeded", supportsHttpsTrafficOnly: true, allowBlobPublicAccess: false, allowSharedKeyAccess: false, minimumTlsVersion: "TLS1_2", publicNetworkAccess: "Enabled", primaryEndpoints: {blob: names.origin + "/"}}}};
      if (storageAccepted && path.startsWith(names.containerId + "?")) return {status: 200, value: {id: names.containerId, properties: {publicAccess: "None"}}};
      if (storageAccepted && path.includes("-transfer?")) return {status: 200, value: {properties: {provisioningState: "Succeeded"}}};
      if (runnerAccepted && path.startsWith(record.deploymentId + "?")) return {status: 200, value: {properties: {provisioningState: "Succeeded"}}};
      if (readinessAccepted && path.includes("/runCommands/")) return {status: 200, value: {properties: {instanceView: {executionState: "Failed", exitCode: 1, error: "Linux bootstrap did not complete successfully."}}}};
      return {status: 404, value: {}};
    },
    runnerList: async (_subscription: string, path: string) => {
      events.push({method: "LIST", path});
      if (path.includes("/skus?")) return [{resourceType: "virtualMachines", name: "Standard_B2s_v2", family: "standardBSv2Family", locations: ["japaneast"], capabilities: [{name: "vCPUs", value: "2"}, {name: "MemoryGB", value: "4"}], locationInfo: [{location: "japaneast", zones: ["1"]}], restrictions: []}];
      if (path.includes("/usages?")) return [{name: {value: "cores"}, currentValue: 0, limit: 10}, {name: {value: "standardBSv2Family"}, currentValue: 0, limit: 10}];
      return [];
    },
    uploadRunnerArchive: async (_r: RunnerRecord, path: string, manifest: unknown) => { events.push({method: "UPLOAD", path, body: manifest}); },
    subscriptions: async () => [{id}, {id: principal}],
    locations: async () => [{name: "japaneast"}],
    storagePrincipal: async () => principal,
    retailRates: async () => [],
    dispose: () => {}
  } as unknown as AzureSession;
  const stages = new LostResponseStages(root, scope, async () => structuredClone(record), actual);
  const control: RunnerControl = {request: stages.request.bind(stages), list: stages.list.bind(stages), sleep: async () => {},
    persist: async r => { record = structuredClone(r); saved.push(structuredClone(r)); events.push({method: "PERSIST:" + r.phase}); }};
  const preview = () => {
    record.artifact = artifact; record.phase = "previewed"; record.hourlyComputeUSD = .1;
    record.template = runnerTemplate(id, record.input, artifact, "ssh-ed25519 AAAA");
    record.expiresAt = new Date(Date.now() + 600000).toISOString();
    record.previewHash = previewHash(record.template, record.input, record.hourlyComputeUSD);
  };
  const storage = async () => {
    record.storageDeployment = storageDraft(record, principal);
    await submitStorage(control, record); await refreshStorage(control, record);
    assert.equal(record.storageDeployment?.phase, "ready");
  };
  return {root, scope, stages, control, events, saved, artifact, names, preview, storage, actual,
    current: () => record, set: (r: RunnerRecord) => { record = structuredClone(r); }, status: (code: number) => {acceptanceStatus = code;} };
}

test("staged companion completes storage/artifact/preflight, withholds one accepted deployment, and reconciles without replay", {skip: process.platform === "win32"}, async t => {
  const f = await fixture(t);
  await f.storage();
  f.current().developmentUpload = {artifact: f.artifact, phase: "prepared"};
  await f.stages.invoke("uploadRunnerArchive", [structuredClone(f.current()), f.scope.archivePath, {file: id, sha256: sha, bytes: 42}]);
  f.current().developmentUpload!.phase = "ready";
  f.preview(); await f.stages.approvePreview();
  const unknown = await submitRunner(f.control, f.current());
  assert.equal(unknown.phase, "unknown");
  assert.equal(f.events.filter(e => e.method === "PUT" && e.path?.startsWith(unknown.deploymentId + "?")).length, 1);
  const putAt = f.events.findIndex(e => e.method === "PUT" && e.path?.startsWith(unknown.deploymentId + "?"));
  assert.equal(f.events[putAt - 1]!.method, "PERSIST:deployment-submitted");
  const receiptText = await readFile(join(f.root, "azure-acceptance.json"), "utf8");
  const receipt = JSON.parse(receiptText);
  assert.equal(receipt.status, 201); assert.equal(receipt.independentARMObservationRequired, true);
  assert.equal(receipt.realServiceQualifiedByThisFile, false); assert.ok(!receiptText.includes("privateDiagnostic"));
  const before = f.events.length;
  const provisioned = await refreshRunner(f.control, unknown);
  assert.equal(provisioned.phase, "provisioned");
  assert.deepEqual(f.events.slice(before).map(e => e.method), ["GET", "PERSIST:provisioned"]);
  await assert.rejects(submitRunner(f.control, provisioned), /stale|submitted/);
  // A reconstructed adapter cannot bypass the create-only retained PUT intent.
  const approved = JSON.parse(await readFile(join(f.root, "native-approved-preview.json"), "utf8")).record as RunnerRecord;
  f.set(approved);
  const restarted = new LostResponseStages(f.root, f.scope, async () => f.current(), f.actual);
  await assert.rejects(restarted.approvePreview(), /EEXIST/);
  f.set({...approved, phase: "deployment-submitted"});
  await assert.rejects(restarted.request(id, `${approved.deploymentId}?api-version=2022-09-01`, "PUT", {properties: {mode: "Incremental", template: approved.template}}));
  assert.equal(f.events.filter(e => e.method === "PUT" && e.path?.startsWith(approved.deploymentId + "?")).length, 1);
});

test("only one exact normal source-free readiness control reaches the inert adapter", {skip: process.platform === "win32"}, async t => {
  const f = await fixture(t); f.preview(); f.current().phase = "provisioned";
  const submitted = await dispatchGuest(f.control, f.current(), {version: 1, workflow: id, operation, action: "ready"});
  const checked = (await reconcileGuest(f.control, submitted)).record;
  assert.equal(checked.guestCommand?.phase, "failed"); assert.equal(checked.guestReady, undefined); assert.equal(checked.readinessReceipts, undefined);
  assert.equal(checked.assessment, undefined); assert.equal(checked.migration, undefined);
  const puts = f.events.filter(e => e.method === "PUT"); assert.equal(puts.length, 1);
  const body = puts[0]!.body as {properties: {source: {script: string}; protectedParameters: {value: string}[]}};
  assert.equal(body.properties.source.script, guestReadinessScript);
  assert.deepEqual(JSON.parse(Buffer.from(body.properties.protectedParameters[0]!.value, "base64").toString()), {version: 1, workflow: id, operation, action: "ready"});
  const before = f.events.length;
  await assert.rejects(dispatchGuest(f.control, checked, {version: 1, workflow: id, operation, action: "inventory", configuration: {}}), /fresh guest readiness/);
  assert.equal(f.events.length, before);
  f.set({...checked, guestCommand: {...checked.guestCommand!, phase: "submitted"}});
  await assert.rejects(f.stages.request(id, puts[0]!.path!, "PUT", puts[0]!.body), /EEXIST/);
  assert.equal(f.events.filter(e => e.method === "PUT").length, 1);
});

test("foreign scope, forbidden mutation, expired window and source state fail before adapter effects", async t => {
  const attempts: ((f: Awaited<ReturnType<typeof fixture>>) => Promise<unknown>)[] = [
    f => f.stages.request(principal, `${f.current().vmId}?api-version=2024-07-01`),
    f => f.stages.request(id, `${group}/providers/Microsoft.Compute/virtualMachines/foreign?api-version=2024-07-01`),
    f => f.stages.request(id, `${f.current().vmId}?api-version=2024-07-01`, "DELETE"),
    f => f.stages.request(id, `${f.current().vmId}?api-version=2024-07-01`, "PATCH", {}),
    f => f.stages.invoke("startSource", []),
    f => { f.scope.expiresAt = "2000-01-01T00:00:00Z"; return f.stages.request(id, `${f.current().vmId}?api-version=2024-07-01`); },
    f => { f.current().sourceDraft = {} as never; return f.stages.request(id, `${f.current().vmId}?api-version=2024-07-01`); },
    f => { f.current().target = {} as never; return f.stages.request(id, `${f.current().vmId}?api-version=2024-07-01`); }
  ];
  for (const attempt of attempts) {
    const f = await fixture(t); f.preview();
    await assert.rejects(attempt(f)); assert.deepEqual(f.events, []);
  }
});

test("template and deployment body mutations cannot borrow native approval", {skip: process.platform === "win32"}, async t => {
  for (const mutation of ["template", "body", "record", "foreign-path"] as const) {
    const f = await fixture(t); f.preview(); await f.stages.approvePreview();
    f.current().phase = "deployment-submitted";
    const body = {properties: {mode: "Incremental", template: structuredClone(f.current().template)}};
    let path = `${f.current().deploymentId}?api-version=2022-09-01`;
    if (mutation === "template") (f.current().template.resources as {location: string}[])[0]!.location = "eastus";
    if (mutation === "body") body.properties.mode = "Complete";
    if (mutation === "record") f.current().hourlyComputeUSD = 5;
    if (mutation === "foreign-path") path = path.replace("deployments/af-", "deployments/foreign-");
    await assert.rejects(f.stages.request(id, path, "PUT", body));
    assert.deepEqual(f.events, []); assert.ok(!(await readdir(f.root)).includes("runner-put-intent.json"));
  }
});

test("wrong artifact metadata and readiness script/payload cannot reach transport", async t => {
  for (const mutation of ["archive", "digest", "bytes", "script", "source-payload"] as const) {
    const f = await fixture(t); f.preview();
    if (["archive", "digest", "bytes"].includes(mutation)) {
      f.current().developmentUpload = {artifact: f.artifact, phase: "prepared"};
      await assert.rejects(f.stages.invoke("uploadRunnerArchive", [f.current(), mutation === "archive" ? "/foreign/archive.tar.gz" : f.scope.archivePath,
        {file: id, sha256: mutation === "digest" ? "0".repeat(64) : sha, bytes: mutation === "bytes" ? 43 : 42}]));
    } else {
      f.current().phase = "provisioned";
      f.current().guestCommand = {id: `${f.current().vmId}/runCommands/af-${operation}`, operation, action: "ready", phase: "submitted", submittedAt: new Date().toISOString()};
      const payload = {version: 1, workflow: id, operation, action: mutation === "source-payload" ? "inventory" : "ready"};
      const body = {location: "japaneast", properties: {source: {script: mutation === "script" ? "echo altered" : guestReadinessScript}, protectedParameters: [{name: "AF_RUNNER_REQUEST", value: Buffer.from(JSON.stringify(payload)).toString("base64")}], timeoutInSeconds: 60, asyncExecution: false}};
      await assert.rejects(f.stages.request(id, `${f.current().guestCommand!.id}?api-version=2024-07-01`, "PUT", body));
    }
    assert.deepEqual(f.events, []);
  }
});

test("a rejected Azure deployment response leaves uncertainty without an acceptance receipt or replay", {skip: process.platform === "win32"}, async t => {
  const f = await fixture(t); await f.storage(); f.preview(); await f.stages.approvePreview(); f.status(403);
  const unknown = await submitRunner(f.control, f.current());
  assert.equal(unknown.phase, "unknown");
  assert.ok(!(await readdir(f.root)).includes("azure-acceptance.json"));
  assert.equal(f.events.filter(e => e.method === "PUT" && e.path?.startsWith(unknown.deploymentId + "?")).length, 1);
  await assert.rejects(submitRunner(f.control, unknown));
});

// Native Windows directory fsync is intentionally not bypassed by the companion.
test("Windows directory durability refusal prevents any staged transport and retains the consumed file", {skip: process.platform !== "win32"}, async t => {
  const f = await fixture(t); f.preview();
  await assert.rejects(f.stages.approvePreview(), {code: "EPERM", syscall: "fsync"});
  assert.deepEqual(f.events, []);
  const path = join(f.root, "native-approved-preview.json"), retained = await readFile(path);
  assert.equal(JSON.parse(retained.toString()).record.id, id);
  await assert.rejects(f.stages.approvePreview(), {code: "EEXIST"});
  assert.deepEqual(await readFile(path), retained);
  const readiness = await fixture(t); readiness.preview(); readiness.current().phase = "provisioned";
  await assert.rejects(dispatchGuest(readiness.control, readiness.current(), {version: 1, workflow: id, operation, action: "ready"}), {code: "EPERM", syscall: "fsync"});
  assert.deepEqual(readiness.events, [], "No transport or normal intent persists without a durable admission event");
});
