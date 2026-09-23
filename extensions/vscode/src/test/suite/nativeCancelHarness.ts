import * as assert from "node:assert/strict";
import * as vscode from "vscode";
import { createHash, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { lstat, mkdir, open, readFile, readdir, realpath } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { Script } from "node:vm";
import { RunnerStore } from "../../guided/runnerStore";
import { RunnerControl } from "../../core/runnerLifecycle";
import { RunnerRecord } from "../../core/runner";
import { assertPreparationWrite, integratedCancellationFixture, IntegratedNativeCancelCase, integratedNativeCancelCases, selectNativeCancelCases } from "../helpers/nativeCancelCatalog";
import { lazyNativeFacade } from "../helpers/nativeCancelFacade";

// Dedicated interactive harness only; not added to the automatic host suite,
// not registered as a release/test command, never substitutes a Cancel result.
const hash = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const extensionRoot = resolve(__dirname, "../../..");
async function createJSON(path: string, value: unknown): Promise<void> {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(JSON.stringify(value, null, 2) + "\n"); await file.sync(); } finally { await file.close(); }
}
async function snapshot(root: string): Promise<Record<string, { sha256: string; mode: number }>> {
  const result: Record<string, { sha256: string; mode: number }> = {};
  for (const name of (await readdir(root)).sort()) {
    const path = join(root, name), info = await lstat(path);
    assert.ok(info.isFile() && !info.isSymbolicLink(), "Fixture store must contain only regular files");
    result[name] = { sha256: hash(await readFile(path)), mode: info.mode & 0o777 };
  }
  return result;
}

async function oneCase(root: string, scenario: IntegratedNativeCancelCase): Promise<string> {
  const caseRoot = join(root, `${scenario.id}-${randomUUID()}`);
  await mkdir(caseRoot, { mode: 0o700 });
  try { return await preparedCase(caseRoot, scenario); }
  catch (failure) {
    await createJSON(join(caseRoot, "failed-case.json"), { scenario: scenario.id, pass: false,
      error: failure instanceof Error ? failure.message : "Unknown fixture failure", completedAt: new Date().toISOString(),
      limitation: "Failed preparation or execution is not native cancellation evidence." });
    throw failure;
  }
}

async function preparedCase(caseRoot: string, scenario: IntegratedNativeCancelCase): Promise<string> {
  const fixture = integratedCancellationFixture(scenario), storeRoot = join(caseRoot, "runner-v2"), store = new RunnerStore(storeRoot);
  await store.write(fixture.record);
  for (const report of fixture.reports) await store.retainReport(fixture.record.id, report.manifest, report.text);
  const reportText = JSON.stringify({ synthetic: true, scope: "native-cancel-only", scenario: scenario.id, workflow: fixture.record.id });
  await store.retainReport(fixture.record.id, { operation: randomUUID(), bytes: Buffer.byteLength(reportText), sha256: hash(reportText) }, reportText);
  const fixturesRoot = join(caseRoot, "inert-files"); await mkdir(fixturesRoot, { mode: 0o700 });
  for (const fixtureFile of fixture.files) {
    assert.match(fixtureFile.name, /^[A-Za-z0-9_.-]+$/);
    const file = await open(join(fixturesRoot, fixtureFile.name), "wx", 0o600);
    try { await file.writeFile(fixtureFile.text); await file.sync(); } finally { await file.close(); }
  }
  const before = await snapshot(storeRoot), originalRecord = await store.read(fixture.record.id);
  let modalBefore: typeof before | undefined;
  const events: Record<string, unknown>[] = [], deniedEffects: string[] = [];
  const deny = (kind: string): never => { deniedEffects.push(kind); throw new Error(`Inert cancellation harness denied ${kind}`); };
  let modalCount = 0, actualCancel = false, pickIndex = 0, inputIndex = 0, preparationWrites = 0;
  const guardedWrite = async (record: RunnerRecord) => {
    assertPreparationWrite(scenario, fixture, originalRecord, record, preparationWrites, modalCount > 0);
    preparationWrites++;
    events.push({ type: "allowlisted-pre-modal-local-observation", before: originalRecord, after: record });
    await store.write(record);
  };
  const guardedStore = new Proxy(store, { get(target, name) {
    if (name === "write") return guardedWrite;
    if (!["list", "read", "readReport", "exclusive"].includes(String(name))) return () => deny(`store.${String(name)}`);
    const method = Reflect.get(target, name); return typeof method === "function" ? method.bind(target) : method;
  } });
  const control: RunnerControl = {
    persist: guardedWrite,
    list: async (_subscription, path) => {
      const resource = path.split("?")[0]!;
      if (!fixture.lists.has(resource)) return deny("unplanned ARM list");
      events.push({ type: "synthetic-ARM-LIST", path }); return structuredClone(fixture.lists.get(resource)!);
    }, sleep: async () => deny("unexpected wait"),
    request: async (_subscription, path, method = "GET") => {
      if (method !== "GET") return deny(`ARM ${method}`);
      const resource = path.split("?")[0]!;
      if (!fixture.responses.has(resource)) return deny("unplanned ARM GET");
      events.push({ type: "synthetic-ARM-GET", path });
      return { status: 200, value: structuredClone(fixture.responses.get(resource)) };
    }
  };
  const azure = new Proxy({
    runnerRequest: control.request, runnerList: control.list,
    runnerAccountBinding: async () => { if (scenario.id !== "A29") return deny("unexpected account binding"); events.push({ type: "synthetic-account-binding" }); return "b".repeat(64); },
    retailRates: async () => {
      if (scenario.id !== "A18") return deny("unexpected pricing");
      events.push({ type: "synthetic-pricing" });
      return [{ serviceName: "Virtual Machines", armSkuName: "Standard_D4s_v5", effectiveStartDate: "2026-01-01T00:00:00Z", hourlyUSD: 1 },
        { serviceName: "Azure Database for PostgreSQL", armSkuName: "Standard_D4ds_v5", effectiveStartDate: "2026-01-01T00:00:00Z", hourlyUSD: 1 }];
    }
  }, { get: (target, name) => name in target ? target[name as keyof typeof target] : () => deny(`Azure adapter ${String(name)}`) });
  const services = new Proxy({}, { get: (_target, name) => () => deny(`source service ${String(name)}`) });
  let receive: ((message: unknown) => Promise<void>) | undefined;
  const posted: Record<string, unknown>[] = [];
  const privateCommands = new Map<string, (...args: unknown[]) => unknown>();
  // Resolve only requested APIs lazily. Spreading VS Code's namespace eagerly
  // invokes proposed-API getters; proxying its frozen original violates proxy
  // invariants. An empty target avoids both without enabling proposed APIs.
  const windowFacade = new Proxy({} as typeof vscode.window, {
    get(_target, name) {
      if (name === "showWarningMessage") return async (message: string, options: vscode.MessageOptions, ...items: string[]) => {
        assert.equal(message, fixture.title); assert.equal(options.modal, true); assert.equal(++modalCount, 1);
        assert.equal(preparationWrites, fixture.preparationWrites);
        modalBefore = await snapshot(storeRoot);
        if (!preparationWrites) assert.deepEqual(modalBefore, before);
        else {
          assert.deepEqual(Object.keys(modalBefore), Object.keys(before));
          for (const name of Object.keys(before).filter(name => name !== `${fixture.record.id}.json`)) assert.deepEqual(modalBefore[name], before[name]);
        }
        events.push({ type: "native-modal-opened", message, options, items, at: new Date().toISOString() });
        await createJSON(join(caseRoot, "modal-opened.json"), { scenario: scenario.id, message, options, items, before: modalBefore,
          preparationWrites, initial: before, syntheticPrerequisites: true });
        // Forward unchanged to the real native API. No automatic click or
        // undefined substitute is allowed. Positive buttons abort the fixture.
        const result = await vscode.window.showWarningMessage(message, options, ...items);
        events.push({ type: "native-modal-result", result: result ?? null, at: new Date().toISOString() });
        if (result !== undefined) return deny("positive approval (Cancel-only fixture)");
        actualCancel = true;
        return undefined;
      };
      if (name === "showQuickPick") return async (items: unknown[]) => {
        let choice: unknown;
        if (fixture.action) choice = items.find(item => item === fixture.action);
        else choice = items[fixture.selectionIndexes[pickIndex++]!];
        assert.ok(choice !== undefined, "Declared synthetic selection must exist");
        events.push({ type: "synthetic-selection", choice }); return choice;
      };
      if (name === "showInputBox") return async (options: vscode.InputBoxOptions) => {
        const value = fixture.inputValues[inputIndex++]; assert.ok(value !== undefined);
        events.push({ type: "synthetic-input", prompt: options.prompt, value }); return value;
      };
      if (name === "showInformationMessage") return async (message: string) => { events.push({ type: "controller-information", message }); return undefined; };
      if (name === "showOpenDialog") return async () => {
        if (!fixture.openFile) return deny("undeclared file selection");
        events.push({ type: "synthetic-file-selection", file: fixture.openFile });
        return [vscode.Uri.file(join(fixturesRoot, fixture.openFile))];
      };
      if (name === "createWebviewPanel") return () => {
        if (scenario.family !== "source" && scenario.id !== "A01" || receive) return deny("unexpected synthetic panel");
        return { onDidDispose: () => ({ dispose() {} }), reveal() {}, webview: { cspSource: "synthetic-native-cancel", html: "",
          onDidReceiveMessage: (listener: typeof receive) => { receive = listener; return { dispose() {} }; },
          postMessage: async (message: Record<string, unknown>) => { posted.push(message); return true; } } };
      };
      if (name === "showSaveDialog" || name === "showErrorMessage" || name === "withProgress") return () => deny(`unexpected UI ${String(name)}`);
      return Reflect.get(vscode.window, name);
    }
  });
  const commandFacade = new Proxy({} as typeof vscode.commands, { get: (_target, name) => {
    if (name === "registerCommand" && scenario.id === "A01") return (id: string, handler: (...args: unknown[]) => unknown) => {
      assert.ok(!privateCommands.has(id)); privateCommands.set(id, handler); return { dispose() {} };
    };
    return () => deny(`unexpected command API ${String(name)}`);
  } });
  const facade = lazyNativeFacade(vscode, { window: windowFacade, commands: commandFacade });
  const outputRoot = resolve(__dirname, "../.."), modulePath = join(outputRoot, `${scenario.module}.js`);
  const code = await readFile(modulePath, "utf8");
  await createJSON(join(caseRoot, "baseline.json"), { scenario, workflow: fixture.record.id, before,
    controller: { path: modulePath, compiledSHA256: hash(code), sourceSHA256: hash(await readFile(join(extensionRoot, `src/${scenario.module}.ts`))) },
    productionBundleSHA256: hash(await readFile(join(extensionRoot, "dist/extension.js"))),
    prerequisiteNote: fixture.prerequisiteNote, preparationWritesAllowed: fixture.preparationWrites,
    evidenceLevel: "compiled-production-controller + actual-native-modal + synthetic-prerequisites + inert-effects" });
  // Load unchanged compiled modules. UI dependencies use the same inert facade;
  // environmental Azure/Store constructors are scoped adapters backed by the
  // real private RunnerStore. No production gate or business function is stubbed.
  const cache = new Map<string, { exports: Record<string, (...args: unknown[]) => unknown> }>();
  const loadedModules: { path: string; sha256: string }[] = [];
  const load = (path: string): Record<string, (...args: unknown[]) => unknown> => {
    const existing = cache.get(path); if (existing) return existing.exports;
    const module = { exports: {} as Record<string, (...args: unknown[]) => unknown> }; cache.set(path, module);
    const code = readFileSync(path, "utf8"), nativeRequire = createRequire(path);
    loadedModules.push({ path, sha256: hash(code) });
    new Script(`(function(require,module,exports,__filename,__dirname){${code}\n})`, { filename: path }).runInThisContext()(
      (name: string) => {
        if (name === "vscode") return facade;
        const resolved = nativeRequire.resolve(name);
        if (resolved === join(outputRoot, "guided/azure.js")) return { AzureSession: class { constructor() { return azure; } } };
        if (resolved === join(outputRoot, "guided/runnerStore.js")) return { ...nativeRequire(name), RunnerStore: class {
          constructor(path: string) { assert.equal(path, storeRoot); return guardedStore; }
        } };
        return name.startsWith(".") && resolved.startsWith(outputRoot + "/") && resolved.endsWith(".js") ? load(resolved) : nativeRequire(name);
      }, module, module.exports, path, dirname(path));
    return module.exports;
  };
  const moduleExports = load(modulePath);
  const context = { subscriptions: [], extension: { packageJSON: { version: "2.4.0" } }, globalStorageUri: vscode.Uri.file(caseRoot),
    secrets: { get: () => deny("secret read"), store: () => deny("secret write"), delete: () => deny("secret delete") } };
  let error: string | undefined;
  try {
    const handler = moduleExports[scenario.exported]; assert.ok(handler);
    // Type references also make tsc emit these production entry modules into
    // out-test; dynamic module-path loading alone would omit their output.
    if (scenario.family === "source") {
      const entry = handler as typeof import("../../runnerSourcePanel").openRunnerSource;
      entry(context as unknown as vscode.ExtensionContext, control, guardedStore, fixture.record.id,
        fixture.requiresServices ? services as Parameters<typeof entry>[4] : undefined);
      assert.ok(receive);
      for (const message of fixture.prepareMessages) await receive(message);
      assert.ok(!posted.some(message => message.kind === "error"));
      await receive(fixture.message);
    } else if (scenario.id === "A01") {
      const entry = handler as typeof import("../../runnerMigration").registerRunnerMigration;
      entry(context as unknown as vscode.ExtensionContext, { info() {}, error() {} } as unknown as vscode.LogOutputChannel);
      assert.ok(privateCommands.has("agefreighter.newGuidedMigration"));
      await privateCommands.get("agefreighter.newGuidedMigration")!();
      assert.ok(receive);
      for (const message of fixture.prepareMessages) await receive(message);
      assert.ok(!posted.some(message => message.kind === "error"));
      await receive(fixture.message);
    } else if (scenario.id === "A02" || scenario.id === "A03") {
      const entry = handler as typeof import("../../developmentRunner").prepareDevelopmentRunner;
      await entry(control, guardedStore, azure as unknown as Parameters<typeof entry>[2]);
    } else if (scenario.id === "A17") {
      const entry = handler as typeof import("../../runnerTargetPanel").reviewRunnerTarget;
      await entry(context as unknown as vscode.ExtensionContext, control, guardedStore, azure as unknown as Parameters<typeof entry>[3], fixture.record.id);
    } else if (scenario.id === "A28") {
      const entry = handler as typeof import("../../p1DiagnosticPanel").diagnoseP1;
      await entry(context as unknown as vscode.ExtensionContext, control, guardedStore, azure as unknown as Parameters<typeof entry>[3], fixture.record.id);
    } else if (scenario.id === "A29") {
      const entry = handler as typeof import("../../runnerReceiptRemovalPanel").manageReadinessRemoval;
      await entry(control, guardedStore, azure as unknown as Parameters<typeof entry>[2]);
    } else {
      const entry = handler as typeof import("../../runnerExecutionPanel").continueRunnerExecution;
      await entry(context as unknown as vscode.ExtensionContext, control, guardedStore, azure as unknown as Parameters<typeof entry>[3], fixture.record.id);
    }
    assert.equal(modalCount, 1); assert.equal(actualCancel, true); assert.deepEqual(deniedEffects, []);
    assert.equal(preparationWrites, fixture.preparationWrites); assert.ok(modalBefore);
    assert.ok(!posted.some(message => message.kind === "error"), "Production listener reported an error");
    assert.deepEqual(await snapshot(storeRoot), modalBefore, "Cancel must preserve every post-review record/report byte");
  } catch (failure) { error = failure instanceof Error ? failure.message : "Unknown fixture failure"; }
  const result = { scenario: scenario.id, name: scenario.name, pass: !error, actualNativeCancel: actualCancel, modalCount, events,
    deniedEffects, before, modalBefore, preparationWrites, posted, loadedModules, after: await snapshot(storeRoot), error: error ?? null,
    cloudRequests: 0, signedInCloudEvidence: false, productionCommandRegistrationTested: false,
    limitation: "Native modal/controller cancellation only. Prerequisites/selector inputs are synthetic. Initial-to-modal changes are separately allowlisted. cloudRequests means fixture business requests, not VS Code background networking. No installed command or signed-in ARM admission claim.",
    completedAt: new Date().toISOString() };
  await createJSON(join(caseRoot, "result.json"), result);
  if (error) throw new Error(`${scenario.id} failed; retained ${caseRoot}: ${error}`);
  return caseRoot;
}

export async function run(extensionMode?: vscode.ExtensionMode): Promise<void> {
  const root = process.env.AF_NATIVE_CANCEL_ROOT;
  assert.ok(root && /^\/private\/tmp\/af-native-cancel-[a-zA-Z0-9]{6}$/.test(root));
  assert.equal(await realpath(root), root);
  const rootInfo = await lstat(root); assert.ok(rootInfo.isDirectory() && !rootInfo.isSymbolicLink());
  assert.equal(rootInfo.mode & 0o777, 0o700);
  assert.equal(vscode.env.appHost, "desktop");
  assert.equal(vscode.extensions.getExtension("rioriost.agefreighter")?.extensionPath, extensionRoot);
  assert.equal(extensionMode, vscode.ExtensionMode.Development,
    "VS Code test mode suppresses native dialogs; use the disposable normal-development companion");
  assert.equal(vscode.workspace.workspaceFolders?.length ?? 0, 0, "Use a folderless disposable profile");
  assert.equal(vscode.workspace.isTrusted, true, "Never override workspace trust for a fixture");
  const cases = selectNativeCancelCases(process.env.AF_NATIVE_CANCEL_CASES ?? "");
  const notice = await vscode.workspace.openTextDocument({ language: "markdown", content:
    `# ISOLATED B09 CANCEL-ONLY FIXTURES\n\nRoot: ${root}\n\nSelected ${cases.length} of the frozen 20 remaining decisions: ${cases.map(s => s.id).join(", ")}.\n\nThese are production controllers with synthetic prerequisites and inert effect adapters. Choose Cancel in each actual native modal. Positive approval aborts the fixture. Source review/status-only local preparation is separately audited. Development artifact inputs are harmless text, never executable. No cloud account, credentials, operator store or installed command is tested.\n` });
  await vscode.window.showTextDocument(notice, { preview: false });
  const completed: { scenario: string; evidence: string }[] = [];
  let failure: unknown;
  for (const scenario of cases) {
    const pick = await vscode.window.showQuickPick([`Open ${scenario.id}: ${scenario.name}`, "Stop fixtures"], {
      title: "ISOLATED B09 — actual native Cancel only", placeHolder: "Synthetic prerequisites; no cloud or normal profile access", ignoreFocusOut: true });
    if (!pick || pick === "Stop fixtures") break;
    try { completed.push({ scenario: scenario.id, evidence: await oneCase(root, scenario) }); }
    catch (error) { failure = error; break; }
  }
  await createJSON(join(root, "ledger.json"), { completed, outcome: failure ? "failed" : completed.length === cases.length ? "completed" : "stopped-incomplete", pass: completed.length === cases.length,
    error: failure instanceof Error ? failure.message : failure ? "Unknown fixture failure" : null,
    unexecuted: cases.filter(s => !completed.some(c => c.scenario === s.id)).map(s => s.id),
    selected: cases.map(s => s.id), excludedFromTrial: integratedNativeCancelCases.filter(s => !cases.some(c => c.id === s.id)).map(s => s.id),
    implementedSubset: integratedNativeCancelCases.map(s => s.id), frozenDecisionCount: 30,
    remainingUncreditedCountBeforeTrial: 20, signedInCloudEvidence: false,
    limitation: "Only listed completed native cancellation fixtures are tested. Excluded or unexecuted cases are not newly credited. No signed-in cloud evidence." });
  if (failure) throw failure;
}
