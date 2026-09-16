import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { Script } from "node:vm";
import { join } from "node:path";
import { transformSync } from "esbuild";
import * as runner from "../../core/runner";
import { requirePanelWorkflow } from "../../core/runnerPanelBinding";

// Execute the production message handler with inert UI/storage/Azure adapters.
// This tests lifecycle/dispatch only, not signed-in or live cloud behavior.
const id = "11111111-1111-4111-8111-111111111111";
const input = {subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: `/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/test/subnets/runner`, source: {type: "neo4j" as const, location: "on-premises" as const}};
const code = transformSync(readFileSync(join(__dirname, "../../runnerMigration.ts"), "utf8"), {loader: "ts", format: "cjs"}).code;

function fixture() {
  const record = runner.sourceWorkflowDraft(id, input);
  record.phase = "provisioned";
  const commands = new Map<string, () => unknown>();
  const writes: runner.RunnerRecord[] = [];
  const panels: {messages: Record<string, any>[]; receive: (m: unknown) => Promise<void>; dispose: () => void}[] = [];
  let effects = 0;
  let fileDialogs = 0;
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
    "./guided/azure": {AzureSession: class { async subscriptions() {return [];} async runnerRequest() {effects++; throw Error("Unexpected cloud request");} async runnerList() {effects++; throw Error("Unexpected cloud list");} }},
    "./guided/runnerStore": {RunnerLockedError: class extends Error {}, RunnerStore: class { async list() {return [record];} async read() {return record;} async write(r: runner.RunnerRecord) {writes.push(r);} async exclusive(_id: string, action: () => unknown) {return action();} }},
    "./core/runnerView": {runnerHTML: () => "synthetic UI adapter"},
    "./core/runner": runner,
    "./core/runnerPanelBinding": {requirePanelWorkflow},
    "./core/runnerLifecycle": {refreshRunner: (control: unknown, r: runner.RunnerRecord) => refresh(control, r)},
    "./core/runnerGuest": {dispatchGuest: () => {effects++; throw Error("Unexpected dispatch");}},
    "./runnerSourcePanel": {openRunnerSource: () => {effects++;}},
    "./runnerTargetPanel": {reviewRunnerTarget: () => {effects++;}},
    "./runnerExecutionPanel": {continueRunnerExecution: () => {effects++;}},
    "./developmentRunner": {}, "./core/runnerPlacement": {}
  };
  const output = {exports: {registerRunnerMigration: (_c: unknown, _o: unknown) => {}}};
  const nativeRequire = createRequire(__filename);
  new Script(code).runInNewContext({Error, module: output, exports: output.exports, require: (name: string) => {
    if (name in modules) return modules[name];
    if (name.startsWith("node:")) return nativeRequire(name);
    throw Error("Unexpected dependency: " + name);
  }});
  output.exports.registerRunnerMigration({subscriptions: [], globalStorageUri: {fsPath: "unused-inert-store"}}, {});
  return {record, panels, writes, fileDialogs: () => fileDialogs, effects: () => effects, setRefresh: (fn: typeof refresh) => {refresh = fn;}, open: () => {commands.get("agefreighter.newGuidedMigration")!(); return panels.at(-1)!;}};
}

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
