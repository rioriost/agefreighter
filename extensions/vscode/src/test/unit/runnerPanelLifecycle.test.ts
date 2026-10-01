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
import * as executionActions from "../../core/runnerExecutionActions";
import * as target from "../../core/runnerTarget";
import type { TargetReviewFeedback } from "../../runnerTargetPanel";
import type { ExecutionProgress } from "../../runnerExecutionPanel";
import { otherCancellationFixture, otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";

// Execute the production message handler with inert UI/storage/Azure adapters.
// This tests lifecycle/dispatch only, not signed-in or live cloud behavior.
const id = "11111111-1111-4111-8111-111111111111";
const input = {subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: `/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/test/subnets/runner`, source: {type: "neo4j" as const, location: "on-premises" as const}};
const code = transformSync(readFileSync(join(__dirname, "../../runnerMigration.ts"), "utf8"), {loader: "ts", format: "cjs"}).code;

function fixture(preview?: {preflightError?: string; checksum?: string; arm?: Pick<RunnerControl, "request" | "list">}) {
  const record = runner.sourceWorkflowDraft(id, input);
  record.phase = "provisioned";
  let stored = record;
  const commands = new Map<string, () => unknown>();
  const writes: runner.RunnerRecord[] = [];
  const panels: {messages: Record<string, any>[]; receive: (m: unknown) => Promise<void>; dispose: () => void}[] = [];
  let effects = 0;
  let fileDialogs = 0;
  const previewSteps: string[] = [];
  const workspace = {isTrusted: true};
  const submitted: runner.RunnerRecord[] = [], requests: Parameters<RunnerControl["request"]>[] = [], executions: unknown[][] = [];
  let developmentOptIn = true, deploymentAdapter = false;
  let confirmation: string | undefined = "Create reviewed runner";
  let duringConfirm = async () => {}, duringPersist = async () => {};
  let targetReview = async (..._args: unknown[]) => {effects++;};
  let execute = async (..._args: unknown[]) => {};
  let nativeError = async (_message: string): Promise<unknown> => undefined;
  const targetErrors: unknown[][] = [], sourceReviews: ((workflow: string) => Promise<void>)[] = [];
  let submit = async (_control: RunnerControl, _r: runner.RunnerRecord): Promise<runner.RunnerRecord> => {effects++; throw Error("Unexpected deployment");};
  let refresh = async (_control: unknown, r: runner.RunnerRecord) => r;
  let watch = async (_control: unknown, _store: unknown, _workflow: string, _cancelled: () => boolean, _progress: (r: runner.RunnerRecord) => Promise<void>) => {};
  let targetWatch = async (..._args: Parameters<typeof watch>) => {};
  const modules: Record<string, unknown> = {
    "vscode": {ViewColumn: {One: 1}, workspace, commands: {registerCommand: (name: string, handler: () => unknown) => {commands.set(name, handler); return {}; }}, window: {
      showQuickPick: async () => ({record: structuredClone(stored)}),
      showWarningMessage: async () => {await duringConfirm(); return confirmation;},
      showOpenDialog: async () => {fileDialogs++; return undefined;},
      showErrorMessage: (message: string) => nativeError(message),
      createWebviewPanel: () => {
        const p = {messages: [] as Record<string, any>[], receive: async (_m: unknown) => {}, dispose: () => {}};
        panels.push(p);
        return {reveal: () => {}, onDidChangeViewState: () => {}, onDidDispose: (fn: () => void) => {p.dispose = fn;}, webview: {cspSource: "test", html: "", postMessage: async (m: Record<string, any>) => {p.messages.push(m);}, onDidReceiveMessage: (fn: typeof p.receive) => {p.receive = fn;} }};
      }
    }},
    "./guided/azure": {AzureSession: class {
      async subscriptions() {return [];}
      async runnerRequest(...args: Parameters<RunnerControl["request"]>) {
        if (preview?.arm) return preview.arm.request(...args);
        if (deploymentAdapter) {requests.push(args); return {status: 200, value: {}};}
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
    "./guided/runnerStore": {RunnerLockedError: class extends Error {}, RunnerStore: class { async list() {return [stored];} async read() {return structuredClone(stored);} async write(r: runner.RunnerRecord) {writes.push(r); stored = r; await duringPersist();} async exclusive(_id: string, action: () => unknown) {return action();} }},
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
      submitRunner: async (control: RunnerControl, r: runner.RunnerRecord) => {submitted.push(r); return submit(control, r);}
    },
    "./core/runnerGuest": {dispatchGuest: () => {effects++; throw Error("Unexpected dispatch");}},
    "./runnerSourcePanel": {openRunnerSource: (...args: unknown[]) => {effects++;assert.equal(typeof args[5], "function");sourceReviews.push(args[5] as (workflow: string) => Promise<void>);}},
    "./runnerTargetPanel": {reviewRunnerTarget: (...args: unknown[]) => targetReview(...args)},
    "./runnerExecutionPanel": {continueRunnerExecution: async (...args: unknown[]) => {executions.push(args);effects++;await execute(...args);}},
    "./core/runnerExecutionActions": executionActions,
    "./core/runnerTarget": target,
    "./sourceCredentialPanel": {},
    "./runnerWatch": {watchRunnerState: (...args: Parameters<typeof watch>) => watch(...args), watchTargetState: (...args: Parameters<typeof watch>) => targetWatch(...args)},
    "./core/runnerSourceReport": {},
    "./runnerReceiptsPanel": {},
    "./runnerReceiptRemovalPanel": {},
    "./runnerLockRecoveryPanel": {},
    "./developmentRunner": {developmentEnabled: () => developmentOptIn}, "./core/runnerPlacement": placement
  };
  const output = {exports: {registerRunnerMigration: (_c: unknown, _o: unknown) => {}}};
  const nativeRequire = createRequire(__filename);
  new Script(code).runInNewContext({Error, Date, AbortSignal, fetch: async () => {
    assert.ok(preview); previewSteps.push("release");
    return {ok: preview.checksum !== undefined, text: async () => preview.checksum};
  }, module: output, exports: output.exports, require: (name: string) => {
    if (name in modules) return modules[name];
    if (name.startsWith("node:")) return nativeRequire(name);
    throw Error("Unexpected dependency: " + name);
  }});
  output.exports.registerRunnerMigration({subscriptions: [], globalStorageUri: {fsPath: "unused-inert-store"}, extension: {packageJSON: {version: "2.4.1"}}}, {error: (...args: unknown[]) => targetErrors.push(args)});
  return {record, panels, writes, previewSteps, workspace, submitted, requests, executions, sourceReviews, targetErrors,
    setTargetReview: (fn: typeof targetReview) => {targetReview = fn;},
    setExecution: (fn: typeof execute) => {execute = fn;},
    setNativeError: (fn: typeof nativeError) => {nativeError = fn;},
    reviewTargetCommand: () => commands.get("agefreighter.reviewRunnerTarget")!(),
    changeStored: (fn: (r: runner.RunnerRecord) => void) => fn(stored),
    setConfirm: (fn: typeof duringConfirm) => {duringConfirm = fn;}, cancel: () => {confirmation = undefined;},
    setPersist: (fn: typeof duringPersist) => {duringPersist = fn;}, disableDevelopment: () => {developmentOptIn = false;},
    setSubmit: (fn: typeof submit) => {submit = fn; deploymentAdapter = true;},
    setWatch: (fn: typeof watch) => {watch = fn;},
    setTargetWatch: (fn: typeof targetWatch) => {targetWatch = fn;},
    fileDialogs: () => fileDialogs, effects: () => effects, setRefresh: (fn: typeof refresh) => {refresh = fn;}, open: () => {commands.get("agefreighter.newGuidedMigration")!(); return panels.at(-1)!;}};
}

test("source target review uses the throwing callback rather than the notification-handling command", async () => {
  const f = fixture(), panel = f.open();
  f.changeStored(r => {r.input = runner.parseRunnerInput(input);});
  await panel.receive({action: "restore"});
  await panel.receive({action: "configureSource", workflow: id, input});
  assert.equal(f.sourceReviews.length, 1, JSON.stringify(panel.messages));
  f.setNativeError(async () => assert.fail("Source errors belong in the source panel"));
  f.setTargetReview(async (...args) => {assert.equal(args[4], id);throw Error("Runner provisioning pending");});
  await assert.rejects(f.sourceReviews[0]!(id), /Runner provisioning pending/);
  assert.equal(f.targetErrors.length, 1);
});

test("palette target review does not wait for its error notification to be dismissed", async () => {
  const f = fixture();
  let dismiss!: () => void, shown = false;
  const notification = new Promise<void>(resolve => {dismiss = resolve;});
  f.setTargetReview(async () => {throw Error("Runner provisioning pending");});
  f.setNativeError(async message => {shown = true;assert.match(message, /provisioning pending/);await notification;});
  let completed = false;
  const pending = Promise.resolve(f.reviewTargetCommand()).then(() => {completed = true;});
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(shown, true);
  assert.equal(completed, true, "native notification must not hold the command open");
  dismiss(); await pending;
});

for (const [step, text] of [["preload", "Restarting PostgreSQL..."], ["resize", "Working... Stopping and deallocating the runner VM..."]] as const)
test(`${step} forwards progress and ignores late updates after stopping its monitor`, async () => {
  const f = fixture(), panel = f.open();
  await panel.receive({action: "restore"});
  f.setExecution(async (...args) => {
    assert.equal(args[5], step);
    const update = args[6] as (record: runner.RunnerRecord, progress?: ExecutionProgress) => Promise<void>;
    const cancelled = args[7] as () => boolean;
    await update(f.record, {text, active: true});
    assert.equal(panel.messages.at(-1)!.text, text);
    assert.equal(panel.messages.at(-1)!.active, true);
    await panel.receive({action: "stopWatch"});
    assert.equal(cancelled(), true);
    const count = panel.messages.length;
    await update(f.record, {text: "Late execution completion", active: false});
    assert.equal(panel.messages.length, count);
  });
  await panel.receive({action: "executionAction", workflow: id, step});
  assert.equal(panel.messages.at(-1)!.kind, "busy"); assert.equal(panel.messages.at(-1)!.value, false);
});

test("reconnecting to a submitted target monitors it and publishes the newly enabled step 5-1", async () => {
  const f = fixture(), panel = f.open();
  f.record.target = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!).record.target;
  f.record.target!.phase = "submitted";
  let watches = 0;
  f.setTargetWatch(async (_c, _s, workflow, stopped, progress) => {
    watches++; assert.equal(workflow, id); assert.equal(stopped(), false);
    f.record.target!.phase = "provisioned"; await progress(f.record);
  });
  await panel.receive({action: "restore"});
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(watches, 1);
  const records = panel.messages.filter(m => m.kind === "record");
  assert.equal(records[0]!.record.execution.actions.preload.enabled, false);
  assert.equal(records.at(-1)!.record.execution.actions.preload.enabled, true);
  assert.match(panel.messages.at(-1)!.text, /5-1.*Migration has not started/);
  assert.equal(panel.messages.at(-1)!.active, false);
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});

for (const boundary of ["stopWatch", "selectionChanged", "dispose", "trust"] as const) test(`target reconnect ignores late progress after ${boundary}`, async () => {
  const f = fixture(), panel = f.open();
  f.record.target = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!).record.target;
  f.record.target!.phase = "submitted";
  let stopped = () => false, update = async (_r: runner.RunnerRecord) => {}, finish = () => {};
  f.setTargetWatch(async (_c, _s, _id, cancelled, progress) => {
    stopped = cancelled; update = progress; await new Promise<void>(resolve => { finish = resolve; });
  });
  await panel.receive({action: "restore"});
  if (boundary === "dispose") panel.dispose();
  else if (boundary === "trust") f.workspace.isTrusted = false;
  else await panel.receive({action: boundary});
  assert.equal(stopped(), true);
  const count = panel.messages.length;
  await update(f.record); finish();
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(panel.messages.length, count + (boundary === "trust" ? 1 : 0));
  if (boundary === "trust") {
    assert.equal(panel.messages.at(-1)!.active, false);
    assert.match(panel.messages.at(-1)!.text, /trust revoked.*Monitoring stopped/);
  }
  assert.equal(f.effects(), 0); assert.equal(f.writes.length, 0);
});

test("immediate target reconnect starts a new generation without accepting the old monitor's result", async () => {
  const f = fixture(), panel = f.open();
  f.record.target = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!).record.target;
  f.record.target!.phase = "submitted";
  const watches: { cancelled: () => boolean; progress: (r: runner.RunnerRecord) => Promise<void>; finish: () => void }[] = [];
  f.setTargetWatch(async (_c, _s, _id, cancelled, progress) => {
    await new Promise<void>(resolve => { watches.push({cancelled, progress, finish: resolve}); });
  });
  await panel.receive({action: "restore"});
  await panel.receive({action: "restore"});
  assert.equal(watches.length, 2); assert.equal(watches[0]!.cancelled(), true); assert.equal(watches[1]!.cancelled(), false);
  const count = panel.messages.length;
  await watches[0]!.progress(f.record); watches[0]!.finish();
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(panel.messages.length, count);
  await panel.receive({action: "accounts"});
  assert.equal(watches.length, 2, "old completion must not clear the current monitor's generation");
  watches[1]!.finish();
  await new Promise<void>(resolve => setImmediate(resolve));
});

test("cancelling an explicit target review does not immediately restart monitoring", async () => {
  const f = fixture(), panel = f.open();
  f.record.target = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!).record.target;
  f.record.target!.phase = "previewed";
  await panel.receive({action: "restore"});
  let watches = 0;
  f.setTargetWatch(async () => { watches++; });
  f.setTargetReview(async (...args) => {
    const feedback = args[5] as TargetReviewFeedback;
    f.record.target!.phase = "submitted";
    await feedback.progress!(f.record, target.targetStatusMessage(f.record), true);
    await panel.receive({action: "stopWatch"});
    assert.equal(feedback.cancelled!(), true);
  });
  await panel.receive({action: "reviewTarget", workflow: id});
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(watches, 0);
  assert.equal(panel.messages.at(-1)!.kind, "busy"); assert.equal(panel.messages.at(-1)!.value, false);
});

test("editing placement cancels the old monitor and ignores its late progress without changing retained work", async () => {
  const f = fixture();
  f.record.guestCommand = { id: "ready", action: "ready", operation: id, phase: "submitted", submittedAt: new Date().toISOString() };
  let stopped = () => false, progress = async (_r: runner.RunnerRecord) => {}, finish = () => {};
  f.setWatch(async (_control, _store, _workflow, cancelled, update) => {
    stopped = cancelled; progress = update; await new Promise<void>(resolve => { finish = resolve; });
  });
  const panel = f.open();
  await panel.receive({ action: "restore" });
  assert.equal(stopped(), false);
  await panel.receive({ action: "selectionChanged" });
  assert.equal(stopped(), true);
  const count = panel.messages.length;
  await progress(f.record); finish();
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(panel.messages.length, count);
  assert.equal(f.writes.length, 0); assert.equal(f.effects(), 0);
});

async function deploymentFixture() {
  const f = fixture();
  Object.assign(f.record, {phase: "previewed", template: {resources: []}, hourlyComputeUSD: 0.05,
    expiresAt: new Date(Date.now() + 60_000).toISOString(), updatedAt: new Date().toISOString(),
    artifact: {version: "2.4.0", url: "inert-fixture", sha256: "a".repeat(64), development: {commit: "abcdef", bytes: 100}}});
  f.record.previewHash = runner.previewHash(f.record.template, f.record.input, f.record.hourlyComputeUSD);
  const panel = f.open();
  await panel.receive({action: "restore"});
  const hash = f.record.previewHash;
  // Only test the panel's controller boundary: no production Azure adapter or
  // cloud connection exists. This inert controller copies intent state just as
  // submitRunner does, so the panel must tolerate its own durable advancement.
  let beforeIntent = async () => {};
  f.setSubmit(async (control, r) => {
    await beforeIntent();
    let next: runner.RunnerRecord = {...r, phase: "deployment-submitted"};
    await control.persist(next);
    try { await control.request(r.input.subscriptionId, `${r.deploymentId}?inert=true`, "PUT", {properties: {template: r.template}}); }
    catch { next = {...next, phase: "unknown"}; await control.persist(next); }
    return next;
  });
  return {...f, panel, beforeIntent: (fn: typeof beforeIntent) => {beforeIntent = fn;},
    deploy: () => panel.receive({action: "deploy", workflow: id, hash, networkApproved: true, costApproved: true})};
}

test("unchanged native deployment approval dispatches once after its own intent write", async () => {
  const f = await deploymentFixture(); await f.deploy();
  assert.equal(f.submitted.length, 1);
  assert.deepEqual(f.writes.map(r => r.phase), ["deployment-submitted"]);
  assert.equal(f.requests.length, 1); assert.equal(f.requests[0]![2], "PUT");
  assert.equal(f.effects(), 0);
});

test("cancelled deployment approval neither submits nor persists", async () => {
  const f = await deploymentFixture(); f.cancel(); await f.deploy();
  assert.equal(f.submitted.length, 0); assert.equal(f.requests.length, 0); assert.equal(f.writes.length, 0);
});

for (const change of ["same-hash-renewal", "revision", "price", "template", "artifact", "source-files", "identity", "already-submitted"] as const) {
  test(`deployment refuses ${change} while its approval modal is open`, async () => {
    const f = await deploymentFixture(), originalHash = f.record.previewHash;
    f.setConfirm(async () => f.changeStored(r => {
      if (change === "same-hash-renewal") r.expiresAt = new Date(Date.parse(r.expiresAt) + 60_000).toISOString();
      if (change === "revision") r.updatedAt = new Date(Date.parse(r.updatedAt) + 1).toISOString();
      if (change === "price") r.hourlyComputeUSD = 0.1;
      if (change === "template") r.template = {resources: [], changed: true};
      if (change === "artifact") r.artifact.sha256 = "b".repeat(64);
      if (change === "source-files") r.sourceFiles = [{id, name: "different.csv", path: "/inert/different.csv"}];
      if (change === "identity") r.id = "22222222-2222-4222-8222-222222222222";
      if (change === "already-submitted") r.phase = "deployment-submitted";
      if (change === "price" || change === "template") r.previewHash = runner.previewHash(r.template, r.input, r.hourlyComputeUSD);
    }));
    await f.deploy();
    if (change === "same-hash-renewal") assert.equal(f.record.previewHash, originalHash);
    assert.equal(f.submitted.length, 0); assert.equal(f.requests.length, 0); assert.equal(f.writes.length, 0);
    assert.match(f.panel.messages.at(-2)!.text, /approved preview changed or expired/);
  });
}

for (const lost of ["trust", "development-opt-in", "panel", "expiry"] as const) {
  test(`deployment refuses ${lost} lost during native confirmation`, async t => {
    const f = await deploymentFixture();
    f.setConfirm(async () => {
      if (lost === "trust") f.workspace.isTrusted = false;
      if (lost === "development-opt-in") f.disableDevelopment();
      if (lost === "panel") f.panel.dispose();
      if (lost === "expiry") t.mock.method(Date, "now", () => Date.parse(f.record.expiresAt));
    });
    await f.deploy();
    assert.equal(f.submitted.length, 0); assert.equal(f.requests.length, 0); assert.equal(f.writes.length, 0);
  });
}

for (const boundary of ["preflight", "intent-persistence"] as const) for (const lost of ["trust", "development-opt-in", "panel", "expiry"] as const) {
  test(`deployment refuses ${lost} lost during awaited ${boundary} before PUT`, async t => {
    const f = await deploymentFixture();
    const change = async () => {
      if (lost === "trust") f.workspace.isTrusted = false;
      if (lost === "development-opt-in") f.disableDevelopment();
      if (lost === "panel") f.panel.dispose();
      if (lost === "expiry") t.mock.method(Date, "now", () => Date.parse(f.record.expiresAt));
    };
    if (boundary === "preflight") f.beforeIntent(change);
    else f.setPersist(change);
    await f.deploy();
    assert.equal(f.submitted.length, 1); // Inert controller entered; no ARM dispatch.
    assert.equal(f.requests.length, 0);
    assert.deepEqual(f.writes.map(r => r.phase), boundary === "preflight" ? [] : ["deployment-submitted", "unknown"]);
  });
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
  for (const action of ["deploy", "refresh", "guestReady", "guestRefresh", "reviewTarget", "continueExecution", "executionAction"]) {
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
  for (const action of ["deploy", "refresh", "guestReady", "guestRefresh", "reviewTarget", "continueExecution", "executionAction"]) {
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

test("direct execution messages bind the workflow and allow only recognized steps", async () => {
  const f = fixture(), panel = f.open(); await panel.receive({ action: "restore" });
  for (const step of [undefined, "delete-all", "__proto__", { action: "start" }]) {
    await panel.receive({ action: "executionAction", workflow: id, step });
    assert.match(panel.messages.at(-2)!.text, /Unsupported migration/);
  }
  await panel.receive({ action: "executionAction", workflow: "foreign", step: "start" });
  assert.equal(f.effects(), 0);
  for (const action of executionActions.executionActions) {
    await panel.receive({ action: "executionAction", workflow: id, step: action.id });
    const args = f.executions.at(-1)!;
    assert.equal(args[4], id); assert.equal(args[5], action.id);
  }
  assert.equal(f.executions.length, executionActions.executionActions.length);
});
