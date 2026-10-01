import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import * as resize from "../../core/runnerResize";
import * as actions from "../../core/runnerExecutionActions";
import * as authorization from "../../core/resizeAuthorization";
import { boundedWatch } from "../../core/boundedWatch";
import type { ExecutionProgress } from "../../runnerExecutionPanel";
import { otherCancellationFixture, otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";

type Stage = "deallocate" | "resize" | "start";
type Controller = (context: unknown, control: RunnerControl, store: unknown, azure: unknown, workflow: string, action: "resize",
  update: (r: RunnerRecord, progress?: ExecutionProgress) => Promise<void>, cancelled: () => boolean) => Promise<void>;

function fixture() {
  let record = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!).record;
  delete record.resize;
  record.targetRestart = {phase: "finished", submittedAt: new Date().toISOString()};
  const initial = structuredClone(record), base = record.vmId.slice(0, record.vmId.indexOf("/providers/"));
  const vm = {location: record.input.region, zones: [record.input.zone],
    tags: {workflow: record.id, application: "agefreighter", purpose: "discovery-and-migration"},
    identity: {type: "SystemAssigned", principalId: record.id, tenantId: record.input.subscriptionId},
    properties: {provisioningState: "Succeeded", hardwareProfile: {vmSize: record.input.size},
      instanceView: {statuses: [{code: "PowerState/running"}]},
      storageProfile: {osDisk: {managedDisk: {id: `${base}/providers/Microsoft.Compute/disks/runner`}}, diskControllerType: "SCSI", dataDisks: []},
      networkProfile: {networkInterfaces: [{id: `${base}/providers/Microsoft.Network/networkInterfaces/runner`}]}}};
  let clock = Date.now(), approved = true, cancelled = false, locked = false, confirmations = 0, prices = 0, readiness = 0;
  let lost: Stage | undefined, progressCloud = true, currentRate = record.target!.input.hourlyUSD;
  let duringPrice = () => {}, duringReadiness = () => {}, duringLock = () => {}, duringSleep = () => {};
  let duringRequest = (_path: string, _method: string) => {}, duringList = (_path: string) => {};
  const token = {isCancellationRequested: false}, workspace = {isTrusted: true};
  const requests: {path: string; method: string; stage?: Stage}[] = [], waits: number[] = [], persisted: RunnerRecord[] = [];
  const updates: {record: RunnerRecord; progress?: ExecutionProgress}[] = [], notifications: string[] = [];
  class Clock extends Date { static override now() { return clock; } }
  const control: RunnerControl = {
    request: async (_s, path, method = "GET", body) => {
      const stage = path === `${initial.vmId}/deallocate?api-version=2024-07-01` ? "deallocate"
        : path === `${initial.vmId}/start?api-version=2024-07-01` ? "start"
        : path === `${initial.vmId}?api-version=2024-07-01` && method === "PATCH" ? "resize" : undefined;
      requests.push({path, method, stage}); duringRequest(path, method);
      if (stage) {
        assert.equal(locked, true); assert.equal(method, stage === "resize" ? "PATCH" : "POST");
        assert.equal(persisted.at(-1)!.resize?.phase, {deallocate: "deallocating", resize: "resizing", start: "starting"}[stage]);
        assert.match(updates.at(-1)!.progress!.text, {deallocate: /Stopping and deallocating/, resize: /Changing the runner VM size/, start: /Starting the resized/}[stage]);
        assert.equal(updates.at(-1)!.progress!.active, true, "phase progress must precede the potentially slow Azure response");
        if (progressCloud) {
          if (stage === "deallocate") vm.properties.instanceView.statuses[0]!.code = "PowerState/deallocated";
          if (stage === "resize") {
            assert.deepEqual(body, {properties: {hardwareProfile: {vmSize: initial.target!.input.loaderSize}}});
            vm.properties.hardwareProfile.vmSize = initial.target!.input.loaderSize;
          }
          if (stage === "start") vm.properties.instanceView.statuses[0]!.code = "PowerState/running";
        }
        if (stage === lost) throw Error("Acknowledgement lost");
        return {status: 202, value: {}};
      }
      assert.equal(method, "GET"); assert.equal(body, undefined);
      if (path === `${initial.vmId}?api-version=2024-07-01&$expand=instanceView`) return {status: 200, value: structuredClone(vm)};
      if (path === `${initial.input.subnetId}?api-version=2024-05-01`) return {status: 200, value: {properties: {delegations: []}}};
      if (path === `${initial.input.subnetId.replace(/\/subnets\/[^/]+$/, "")}?api-version=2024-05-01`) return {status: 200, value: {location: initial.input.region}};
      if (path === `${base}?api-version=2021-04-01`) return {status: 200, value: {}};
      throw Error("Unexpected Azure path: " + path);
    },
    list: async (_s, path) => {
      duringList(path);
      if (path.includes("/skus?")) return [{name: initial.target!.input.loaderSize, resourceType: "virtualMachines", family: "standardDSv5Family",
        locations: [initial.input.region], locationInfo: [{location: initial.input.region, zones: [initial.input.zone]}],
        capabilities: [{name: "vCPUs", value: "4"}, {name: "MemoryGB", value: "16"}], restrictions: []}];
      if (path.includes("/usages?")) return ["cores", "standardDSv5Family"].map(value => ({name: {value}, currentValue: 0, limit: 64}));
      throw Error("Unexpected Azure list");
    },
    persist: async r => { record = structuredClone(r); persisted.push(structuredClone(r)); },
    sleep: async ms => { assert.equal(locked, false); assert.equal(ms, 15000); waits.push(ms); clock += ms; duringSleep(); }
  };
  const store = {read: async () => structuredClone(record),
    exclusive: async <T>(_id: string, run: () => Promise<T>) => {
      assert.equal(locked, false); duringLock(); locked = true;
      try { return await run(); } finally { locked = false; }
    }};
  const modules: Record<string, unknown> = {
    vscode: {workspace, ProgressLocation: {Notification: 1}, window: {
      showQuickPick: () => assert.fail("No second action picker or Start action is allowed"),
      showWarningMessage: async () => { confirmations++; return approved ? "Approve this step" : undefined; },
      withProgress: async (options: {title: string; cancellable: boolean}, run: (p: {report: (m: {message: string}) => void}, t: typeof token) => Promise<void>) => {
        assert.match(options.title, /5-2.*Working/); assert.equal(options.cancellable, true);
        return run({report: ({message}) => { notifications.push(message); }}, token);
      },
      showInformationMessage: () => assert.fail("Direct steps use inline feedback")
    }},
    "./core/runnerResize": resize,
    "./core/runnerExecutionActions": actions,
    "./core/resizeAuthorization": {
      resizeAuthorized: (r: RunnerRecord) => authorization.resizeAuthorized(r, clock),
      authorizeResize: (r: RunnerRecord) => authorization.authorizeResize(r, clock)
    },
    "./core/boundedWatch": {boundedWatch: <T>(options: Parameters<typeof boundedWatch<T>>[0]) => boundedWatch({...options, now: () => clock})},
    "./core/runnerTargetPreflight": {targetComputeRate: () => currentRate},
    "./core/runnerAssessment": {ensureAssessmentReadiness: async (_c: unknown, r: RunnerRecord, stopped: () => boolean) => {
      assert.equal(locked, true); readiness++;
      assert.match(updates.at(-1)!.progress!.text, /Working.*Checking Linux guest readiness/);
      assert.equal(stopped(), false); duringReadiness(); return r;
    }}
  };
  for (const name of ["./core/runnerExecution", "./core/report", "./core/runnerTarget", "./core/runnerDiagnostic", "./p1QualificationPanel", "./p1DiagnosticPanel",
    "./core/runnerSource", "./core/runnerResume", "./migrationVerificationPanel", "./sourceCredentialPanel", "./runnerWatch", "./runnerReportFlow", "./runnerTargetInputs", "./core/runnerGuest"]) modules[name] = {};
  const output = {exports: {} as {continueRunnerExecution: Controller}}, native = createRequire(__filename);
  const code = transformSync(readFileSync(join(__dirname, "../../runnerExecutionPanel.ts"), "utf8"), {loader: "ts", format: "cjs"}).code;
  new Script(code).runInNewContext({module: output, exports: output.exports, Error, Date: Clock, Buffer,
    require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : assert.fail("Unexpected dependency " + name)});
  return {initial, vm, workspace, token, requests, waits, persisted, updates, notifications,
    record: () => record, counts: () => ({confirmations, prices, readiness}), cancel: () => { cancelled = true; },
    deny: () => { approved = false; }, freezeCloud: () => { progressCloud = false; },
    lose: (stage?: Stage) => { lost = stage; }, advance: (ms: number) => { clock += ms; },
    duringPrice: (fn: () => void) => { duringPrice = fn; }, duringReadiness: (fn: () => void) => { duringReadiness = fn; },
    duringLock: (fn: () => void) => { duringLock = fn; }, duringSleep: (fn: () => void) => { duringSleep = fn; },
    duringRequest: (fn: typeof duringRequest) => { duringRequest = fn; }, duringList: (fn: typeof duringList) => { duringList = fn; },
    changeRate: () => { currentRate++; },
    run: () => output.exports.continueRunnerExecution({secrets: {get: () => assert.fail("Resize must not retrieve credentials")}}, control, store,
      {retailRates: async () => {
        prices++; assert.match(updates.at(-1)!.progress!.text, /Working.*Checking current prices/);
        assert.equal(updates.at(-1)!.progress!.active, true); duringPrice(); return [];
      }}, record.id, "resize", async (r, progress) => { updates.push({record: structuredClone(r), progress}); }, () => cancelled)
  };
}

test("one approval automatically runs deallocate, resize and start with immediate working and phase feedback", async () => {
  const f = fixture();
  await f.run();
  assert.deepEqual(f.counts(), {confirmations: 1, prices: 1, readiness: 1});
  assert.deepEqual(f.requests.filter(r => r.stage).map(r => r.stage), ["deallocate", "resize", "start"]);
  assert.deepEqual([...new Set(f.updates.map(u => u.record.resize?.phase).filter(Boolean))],
    ["deallocating", "ready-to-resize", "resizing", "ready-to-start", "starting", "finished"]);
  assert.equal(f.record().resize?.phase, "finished"); assert.equal(f.record().guestReady, undefined);
  assert.equal(f.record().migration, undefined); assert.equal(f.record().input.size, f.initial.input.size);
  assert.equal(f.waits.length, 5);
  assert.ok(f.updates.filter(u => u.record.resize && u.record.resize.phase !== "finished")
    .every(u => !actions.executionActionState(u.record, "readiness").enabled));
  assert.equal(actions.executionActionState(f.record(), "readiness").enabled, true);
  assert.equal(actions.executionActionState(f.record(), "start").enabled, false);
  const last = f.updates.filter(u => u.progress).at(-1)!.progress!;
  assert.equal(last.active, false); assert.match(last.text, /resize complete.*5-3/);
});

test("cancelled approval never starts pricing, readiness, monitoring or any Azure operation", async () => {
  const f = fixture(); f.deny(); await f.run();
  assert.deepEqual(f.counts(), {confirmations: 1, prices: 0, readiness: 0});
  assert.equal(f.requests.length, 0); assert.equal(f.persisted.length, 0); assert.equal(f.updates.length, 0);
});

test("an already correctly sized running VM completes without deallocation, resize or restart", async () => {
  const f = fixture();
  f.vm.properties.hardwareProfile.vmSize = f.initial.target!.input.loaderSize;
  await f.run();
  assert.equal(f.record().resize?.phase, "finished");
  assert.equal(f.requests.filter(r => r.stage).length, 0);
  assert.equal(f.waits.length, 0);
  assert.match(f.notifications.at(-1)!, /resize complete.*5-3/);
});

for (const boundary of ["native", "price", "lock", "readiness", "request", "sleep", "trust"] as const) test(`resize cancellation at ${boundary} prevents further automatic writes`, async () => {
  const f = fixture();
  if (boundary === "native") f.token.isCancellationRequested = true;
  if (boundary === "price") f.duringPrice(f.cancel);
  if (boundary === "lock") f.duringLock(f.cancel);
  if (boundary === "readiness") f.duringReadiness(() => { f.token.isCancellationRequested = true; });
  if (boundary === "request") f.duringRequest((_path, method) => { if (method === "GET") f.token.isCancellationRequested = true; });
  if (boundary === "sleep") f.duringSleep(f.cancel);
  if (boundary === "trust") f.duringPrice(() => { f.workspace.isTrusted = false; });
  await f.run();
  assert.equal(f.requests.filter(r => r.stage).length, boundary === "sleep" ? 1 : 0);
  assert.equal(f.updates.filter(u => u.progress).at(-1)!.progress!.active, false);
  assert.match(f.notifications.at(-1)!, /monitoring stopped/);
  assert.equal(f.record().migration, undefined);
});

for (const stage of ["deallocate", "resize", "start"] as const) test(`unknown ${stage} acknowledgement pauses rather than replaying and can explicitly reconcile`, async () => {
  const f = fixture(); f.lose(stage);
  await f.run();
  assert.equal(f.record().resize?.unknown, true);
  assert.equal(f.requests.filter(r => r.stage === stage).length, 1);
  assert.equal(f.updates.filter(u => u.progress).at(-1)!.progress!.active, false);
  assert.match(f.notifications.at(-1)!, /uncertain.*paused.*5-7/);
  f.lose(); await f.run();
  assert.equal(f.record().resize?.phase, "finished"); assert.equal(f.counts().confirmations, 1);
  assert.deepEqual(f.requests.filter(r => r.stage).map(r => r.stage), ["deallocate", "resize", "start"]);
});

test("an unconfirmed deallocate request remains read-only on explicit continuation", async () => {
  const f = fixture(); f.freezeCloud(); f.lose("deallocate");
  await f.run(); const before = f.requests.length;
  await f.run();
  assert.equal(f.requests.length, before + 1);
  assert.equal(f.requests.filter(r => r.stage).length, 1);
  assert.equal(f.record().resize?.unknown, true);
});

test("slow price/readiness checks are visibly working and price changes block before mutation", async () => {
  const f = fixture(); f.changeRate();
  await assert.rejects(f.run(), /Compute price changed/);
  assert.match(f.notifications[0]!, /Working.*Checking current prices/);
  assert.equal(f.persisted.length, 0); assert.equal(f.requests.length, 0);
  const failed = fixture(); failed.duringReadiness(() => { throw Error("Linux guest is not idle"); });
  await assert.rejects(failed.run(), /not idle/);
  assert.match(failed.notifications.at(-1)!, /Working.*Checking Linux guest readiness/);
  assert.equal(failed.requests.length, 0);
});

test("pending resize is bounded to its authorization and requires renewed approval after expiry", async () => {
  const f = fixture(); f.freezeCloud(); await f.run();
  assert.equal(f.record().resize?.phase, "deallocating");
  assert.equal(f.waits.length, 79);
  assert.equal(f.requests.filter(r => r.stage).length, 1);
  assert.match(f.notifications.at(-1)!, /authorization or time limit.*Use 5-2/);
  f.advance(15000); f.deny(); await f.run();
  assert.equal(f.counts().confirmations, 2); assert.equal(f.requests.filter(r => r.stage).length, 1);
});

test("authorization expiry during a capacity check blocks a new write", async () => {
  const f = fixture();
  f.duringList(path => { if (path.includes("/usages?")) f.advance(20 * 60000); });
  await assert.rejects(f.run(), /authorization expired or changed/);
  assert.equal(f.requests.filter(r => r.stage).length, 0); assert.equal(f.record().resize, undefined);
});

for (const field of ["vm", "artifact", "input", "target", "price"] as const) test(`changed ${field} scope after approval cannot start resizing`, async () => {
  const f = fixture();
  f.duringPrice(() => {
    if (field === "vm") f.record().vmId += "-changed";
    if (field === "artifact") f.record().artifact.sha256 = "f".repeat(64);
    if (field === "input") f.record().input.zone = "2";
    if (field === "target") f.record().target!.serverId += "-changed";
    if (field === "price") f.record().target!.input.budgetUSD++;
  });
  await assert.rejects(f.run(), /scope changed/);
  assert.equal(f.requests.length, 0); assert.equal(f.persisted.length, 0);
});

test("a changed persistent VM identity halts the sequence without resizing or starting", async () => {
  const f = fixture();
  f.duringSleep(() => { f.vm.identity.principalId = "different"; });
  await assert.rejects(f.run(), /identity or security changed/);
  assert.deepEqual(f.requests.filter(r => r.stage).map(r => r.stage), ["deallocate"]);
});
