import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { Script } from "node:vm";
import { join } from "node:path";
import { transformSync } from "esbuild";
import * as runner from "../../core/runner";
import { requirePanelWorkflow } from "../../core/runnerPanelBinding";
import * as placement from "../../core/runnerPlacement";
import { preflightRunner, RunnerControl } from "../../core/runnerLifecycle";

// Execute the production message handler with inert UI/storage/Azure adapters.
// This tests lifecycle/dispatch only, not signed-in or live cloud behavior.
const id = "11111111-1111-4111-8111-111111111111";
const input = {subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: `/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/test/subnets/runner`, source: {type: "neo4j" as const, location: "on-premises" as const}};
const code = transformSync(readFileSync(join(__dirname, "../../runnerMigration.ts"), "utf8"), {loader: "ts", format: "cjs"}).code;

function fixture(preview?: {preflightError?: string; checksum?: string; arm?: Pick<RunnerControl, "request" | "list">}) {
  const record = runner.sourceWorkflowDraft(id, input);
  record.phase = "provisioned";
  const commands = new Map<string, () => unknown>();
  const writes: runner.RunnerRecord[] = [];
  const panels: {messages: Record<string, any>[]; receive: (m: unknown) => Promise<void>; dispose: () => void}[] = [];
  let effects = 0;
  let fileDialogs = 0;
  const previewSteps: string[] = [];
  let refresh = async (_control: unknown, r: runner.RunnerRecord) => r;
  const modules: Record<string, unknown> = {
    "vscode": {ViewColumn: {One: 1}, workspace: {isTrusted: true}, commands: {registerCommand: (name: string, handler: () => unknown) => {commands.set(name, handler); return {}; }}, window: {
      showQuickPick: async () => ({record}),
      showOpenDialog: async () => {fileDialogs++; return undefined;},
      createWebviewPanel: () => {
        const p = {messages: [] as Record<string, any>[], receive: async (_m: unknown) => {}, dispose: () => {}};
        panels.push(p);
        return {reveal: () => {}, onDidDispose: (fn: () => void) => {p.dispose = fn;}, webview: {cspSource: "test", html: "", postMessage: async (m: Record<string, any>) => {p.messages.push(m);}, onDidReceiveMessage: (fn: typeof p.receive) => {p.receive = fn;} }};
      }
    }},
    "./guided/azure": {AzureSession: class {
      async subscriptions() {return [];}
      async runnerRequest(...args: Parameters<RunnerControl["request"]>) {
        if (preview?.arm) return preview.arm.request(...args);
        effects++; throw Error("Unexpected cloud request");
      }
      async runnerList(_subscription: string, path: string) {
        if (preview && path.includes("/resourcegroups?")) return [{name: "test"}];
        if (preview?.arm) return preview.arm.list(_subscription, path);
        effects++; throw Error("Unexpected cloud list");
      }
      async locations() {assert.ok(preview); return [{name: "japaneast", displayName: "Japan East"}];}
      async retailRates() {previewSteps.push("pricing"); throw Error("Pricing sentinel; no what-if");}
    }},
    "./guided/runnerStore": {RunnerLockedError: class extends Error {}, RunnerStore: class { async list() {return [record];} async read() {return record;} async write(r: runner.RunnerRecord) {writes.push(r);} async exclusive(_id: string, action: () => unknown) {return action();} }},
    "./core/runnerView": {runnerHTML: () => "synthetic UI adapter"},
    "./core/runner": runner,
    "./core/runnerPanelBinding": {requirePanelWorkflow},
    "./core/runnerLifecycle": {
      refreshRunner: (control: unknown, r: runner.RunnerRecord) => refresh(control, r),
      preflightRunner: async (control: RunnerControl, selection: runner.RunnerInput) => {
        assert.ok(preview); previewSteps.push("preflight");
        if (preview.arm) return preflightRunner(control, selection);
        if (preview.preflightError) throw Error(preview.preflightError);
      },
      whatIfRunner: async () => {effects++; throw Error("Unexpected what-if");},
      submitRunner: async () => {effects++; throw Error("Unexpected deployment");}
    },
    "./core/runnerGuest": {dispatchGuest: () => {effects++; throw Error("Unexpected dispatch");}},
    "./runnerSourcePanel": {openRunnerSource: () => {effects++;}},
    "./runnerTargetPanel": {reviewRunnerTarget: () => {effects++;}},
    "./runnerExecutionPanel": {continueRunnerExecution: () => {effects++;}},
    "./sourceCredentialPanel": {},
    "./runnerReceiptsPanel": {},
    "./runnerReceiptRemovalPanel": {},
    "./runnerLockRecoveryPanel": {},
    "./developmentRunner": {}, "./core/runnerPlacement": placement
  };
  const output = {exports: {registerRunnerMigration: (_c: unknown, _o: unknown) => {}}};
  const nativeRequire = createRequire(__filename);
  new Script(code).runInNewContext({Error, AbortSignal, fetch: async () => {
    assert.ok(preview); previewSteps.push("release");
    return {ok: preview.checksum !== undefined, text: async () => preview.checksum};
  }, module: output, exports: output.exports, require: (name: string) => {
    if (name in modules) return modules[name];
    if (name.startsWith("node:")) return nativeRequire(name);
    throw Error("Unexpected dependency: " + name);
  }});
  output.exports.registerRunnerMigration({subscriptions: [], globalStorageUri: {fsPath: "unused-inert-store"}, extension: {packageJSON: {version: "2.4.0"}}}, {});
  return {record, panels, writes, previewSteps, fileDialogs: () => fileDialogs, effects: () => effects, setRefresh: (fn: typeof refresh) => {refresh = fn;}, open: () => {commands.get("agefreighter.newGuidedMigration")!(); return panels.at(-1)!;}};
}

test("preview propagates placement rejection before fetching release or allowing effects", async () => {
  const f = fixture({preflightError: "The selected subnet does not exist."}), panel = f.open();
  await panel.receive({action: "preview", input});
  assert.deepEqual(f.previewSteps, ["preflight"]);
  assert.match(panel.messages.at(-2)!.text, /subnet does not exist/);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
  assert.ok(!panel.messages.some(m => m.kind === "record"));
});

test("preview still rejects invalid catalog selection before preflight or release", async () => {
  const f = fixture({}), panel = f.open();
  await panel.receive({action: "preview", input: {...input, region: "madeupregion"}});
  assert.deepEqual(f.previewSteps, []);
  assert.match(panel.messages.at(-2)!.text, /current subscription list/);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});

for (const checksum of [undefined, "invalid-checksum"]) {
  test(`valid placement cannot bypass ${checksum === undefined ? "missing" : "invalid"} release protection`, async () => {
    const f = fixture({checksum}), panel = f.open();
    await panel.receive({action: "preview", input});
    assert.deepEqual(f.previewSteps, ["preflight", "release"]);
    assert.match(panel.messages.at(-2)!.text, /release\/checksums are not available|checksum is missing or ambiguous/);
    assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
    assert.ok(!panel.messages.some(m => m.kind === "record"));
  });
}

test("pricing is reached only after placement and a matching release checksum", async () => {
  const f = fixture({checksum: `${"a".repeat(64)}  agefreighter_v2.4.0_linux_amd64.tar.gz`}), panel = f.open();
  await panel.receive({action: "preview", input});
  assert.deepEqual(f.previewSteps, ["preflight", "release", "pricing"]);
  assert.match(panel.messages.at(-2)!.text, /Pricing sentinel/);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});

// Synthetic ARM responses, but the real message handler, input validation,
// capability parsers and preflight are connected. No Azure session or GUI
// qualification is claimed. Unexpected paths/methods fail closed in this adapter.
function placementARM() {
  const base = `/subscriptions/${id}/resourceGroups/test`;
  const sourceId = `${base}/providers/Microsoft.Compute/virtualMachines/source`;
  const family = "standardBsv2Family";
  const data = {
    skus: [{resourceType: "virtualMachines", name: input.size, family, locations: [input.region],
      capabilities: [{name: "vCPUs", value: "2"}, {name: "MemoryGB", value: "8"}],
      locationInfo: [{location: input.region, zones: ["1", "2", "3"]}], restrictions: [] as unknown[]}],
    quota: [{name: {value: "cores"}, currentValue: 8, limit: 10},
      {name: {value: family}, currentValue: 8, limit: 10}],
    source: {location: input.region, zones: [] as string[], properties: {}}
  };
  const reads: string[] = [];
  const arm: Pick<RunnerControl, "request" | "list"> = {
    async request(subscription, path, method = "GET", body) {
      assert.equal(subscription, id); assert.equal(method, "GET"); assert.equal(body, undefined);
      reads.push(path);
      if (path === `${input.subnetId}?api-version=2024-05-01`) return {status: 200, value: {properties: {delegations: []}}};
      if (path === `${input.subnetId.replace(/\/subnets\/runner$/, "")}?api-version=2024-05-01`) return {status: 200, value: {location: input.region}};
      if (path === `${base}?api-version=2021-04-01`) return {status: 200, value: {}};
      if (path === `${sourceId}?api-version=2024-07-01`) return {status: 200, value: data.source};
      throw Error("Unexpected fixture request");
    },
    async list(subscription, path) {
      assert.equal(subscription, id); reads.push(path);
      if (path === `/subscriptions/${id}/providers/Microsoft.Compute/skus?api-version=2021-07-01&$filter=${encodeURIComponent("location eq 'japaneast'")}`) return data.skus;
      if (path === `/subscriptions/${id}/providers/Microsoft.Compute/locations/japaneast/usages?api-version=2025-04-01`) return data.quota;
      throw Error("Unexpected fixture list");
    }
  };
  return {data, arm, reads, sourceId};
}

const deniedPlacements: {name: string; error: RegExp; reads: number; change: (data: ReturnType<typeof placementARM>["data"]) => void}[] = [
  {name: "SKU absent", error: /SKU is not available/, reads: 4, change: d => {d.skus = []; }},
  {name: "SKU location restricted", error: /SKU is not available/, reads: 4, change: d => {d.skus[0]!.restrictions = [{type: "Location", restrictionInfo: {locations: [input.region]}}]; }},
  {name: "selected zone restricted", error: /SKU is not available/, reads: 4, change: d => {d.skus[0]!.restrictions = [{type: "Zone", restrictionInfo: {locations: [input.region], zones: ["1"]}}]; }},
  {name: "regional quota short by one core", error: /quota is insufficient/, reads: 5, change: d => {d.quota[0]!.currentValue = 9; }},
  {name: "family quota short by one core", error: /quota is insufficient/, reads: 5, change: d => {d.quota[1]!.currentValue = 9; }},
  {name: "regional quota absent", error: /quota is insufficient/, reads: 5, change: d => {d.quota.shift(); }},
  {name: "family quota absent", error: /quota is insufficient/, reads: 5, change: d => {d.quota.pop(); }},
  {name: "malformed quota value", error: /quota is insufficient/, reads: 5, change: d => {d.quota[0]!.limit = Number.NaN; }}
];
for (const scenario of deniedPlacements) test(`production placement-to-panel contract: ${scenario.name}`, async () => {
  const a = placementARM(); scenario.change(a.data);
  const f = fixture({arm: a.arm}), panel = f.open();
  await panel.receive({action: "preview", input});
  assert.deepEqual(f.previewSteps, ["preflight"]);
  assert.equal(a.reads.length, scenario.reads);
  assert.match(panel.messages.at(-2)!.text, scenario.error);
  assert.equal(panel.messages.at(-1)!.kind, "busy");
  assert.equal(panel.messages.at(-1)!.value, false);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
  assert.ok(!panel.messages.some(m => m.kind === "record"));
});

for (const scenario of ["exact quota boundary", "other zone restricted", "other region restricted", "unknown source zone reviewed"])
  test(`production placement-to-panel control: ${scenario}`, async () => {
    const a = placementARM();
    if (scenario === "other zone restricted") a.data.skus[0]!.restrictions = [{type: "Zone", restrictionInfo: {locations: [input.region], zones: ["2"]}}];
    if (scenario === "other region restricted") a.data.skus[0]!.restrictions = [{type: "Location", restrictionInfo: {locations: ["japanwest"]}}];
    const selection = scenario === "unknown source zone reviewed"
      ? {...input, source: {type: "neo4j" as const, location: "azure" as const, resourceId: a.sourceId}} : input;
    const f = fixture({arm: a.arm}), panel = f.open();
    await panel.receive({action: "preview", input: selection});
    assert.deepEqual(f.previewSteps, ["preflight", "release"], JSON.stringify(panel.messages));
    assert.equal(a.reads.length, scenario === "unknown source zone reviewed" ? 6 : 5);
    assert.match(panel.messages.at(-2)!.text, /release\/checksums are not available/);
    assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
    assert.ok(!panel.messages.some(m => m.kind === "record"));
  });

test("unknown source zone cannot reach ARM or release without an explicit runner zone", async () => {
  const a = placementARM(), f = fixture({arm: a.arm}), panel = f.open();
  await panel.receive({action: "preview", input: {...input, zone: "", source: {type: "neo4j", location: "azure", resourceId: a.sourceId}}});
  assert.deepEqual(f.previewSteps, []); assert.deepEqual(a.reads, []);
  assert.match(panel.messages.at(-2)!.text, /zone/);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});

test("canceling the new-wizard CSV picker neither persists nor uploads nor starts a runner", async () => {
  const f = fixture(), panel = f.open();
  await panel.receive({action: "csv"});
  assert.equal(f.fileDialogs(), 1);
  assert.equal(f.effects(), 0);
  assert.equal(f.writes.length, 0);
  assert.ok(!panel.messages.some(m => ["csv", "record", "error"].includes(m.kind)));
});

test("closing and opening a new wizard does not select the retained job", async () => {
  const f = fixture(), old = f.open();
  await old.receive({action: "restore"});
  assert.ok(old.messages.some(m => m.kind === "record" && m.record.id === id));
  old.dispose();
  const next = f.open();
  await next.receive({action: "ready"});
  assert.ok(!next.messages.some(m => m.kind === "record" || m.kind === "restoreInput"));
  for (const action of ["deploy", "refresh", "guestReady", "guestRefresh", "reviewTarget", "continueExecution"]) {
    await next.receive({action, workflow: id});
    assert.ok(next.messages.some(m => m.kind === "error" && /Reconnect/.test(m.text)));
  }
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});

test("late work in a disposed panel cannot post to or block the next panel", async () => {
  const f = fixture(), old = f.open(); await old.receive({action: "restore"});
  let finish!: () => void;
  const waiting = new Promise<void>(resolve => {finish = resolve;});
  f.setRefresh(async (_control, r) => {await waiting; return r;});
  const pending = old.receive({action: "refresh", workflow: id});
  old.dispose(); const next = f.open(); await next.receive({action: "ready"});
  assert.ok(next.messages.some(m => m.kind === "subscriptions"));
  const before = next.messages.length; finish(); await pending;
  assert.equal(next.messages.length, before);
  assert.ok(!next.messages.some(m => m.kind === "record"));
  const oldCount = old.messages.length; await old.receive({action: "restore"});
  assert.equal(old.messages.length, oldCount);
});

test("invalidated messages cannot fall back to host-side current workflow", async () => {
  const f = fixture(), panel = f.open(); await panel.receive({action: "restore"});
  for (const action of ["deploy", "refresh", "guestReady", "guestRefresh", "reviewTarget", "continueExecution"]) {
    await panel.receive({action});
    assert.ok(panel.messages.at(-2)?.kind === "error");
  }
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
  await panel.receive({action: "configureSource", workflow: id, input: {...input, region: "japanwest"}});
  assert.match(panel.messages.at(-2)!.text, /changed/);
  assert.equal(f.effects(), 0);
});

test("account refresh restores fields before presenting saved readiness", async () => {
  const f = fixture(), panel = f.open(); await panel.receive({action: "restore"});
  const start = panel.messages.length; await panel.receive({action: "accounts"});
  assert.deepEqual(panel.messages.slice(start).map(m => m.kind), ["busy", "subscriptions", "restoreInput", "record", "busy"]);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});
