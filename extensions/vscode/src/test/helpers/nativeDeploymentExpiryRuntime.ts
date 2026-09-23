/** Loaded only by a separately prepared disposable normal development host.
 * Production logic is bundled unchanged except test-only observation facades.
 * No Azure adapter, account lookup or network transport is provided. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, open, readFile, readdir, realpath } from "node:fs/promises";
import { join } from "node:path";
import type * as VSCode from "vscode";
import { registerRunnerMigration } from "../../runnerMigration";
import { RunnerStore } from "../../guided/runnerStore";
import { otherCancellationFixture, otherNativeCancelCases } from "./nativeCancelOtherScenarios";
import { lazyNativeFacade } from "./nativeCancelFacade";
import { DeploymentExpiryObservation, deploymentExpiryFailures } from "./nativeDeploymentExpiry";

interface ExpiryBridge { vscode: typeof VSCode; azure: object; denyFetch: () => never; submitEntered: () => void; storeWrite: () => void }
declare global { var __afDeploymentExpiry: ExpiryBridge; }
const actual = module.require("vscode") as typeof VSCode;
const hash = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
async function retain(path: string, value: unknown) {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(JSON.stringify(value, null, 2) + "\n"); await file.sync(); } finally { await file.close(); }
}
async function snapshot(root: string) {
  const result: Record<string, string> = {};
  for (const name of (await readdir(root)).sort()) {
    const path = join(root, name), info = await lstat(path);
    assert.ok(info.isFile() && !info.isSymbolicLink());
    result[name] = `${info.mode & 0o777}:${hash(await readFile(path))}`;
  }
  return result;
}

export async function activate(context: VSCode.ExtensionContext) {
  const root = context.extensionPath;
  assert.match(root, /^\/private\/tmp\/af-deployment-expiry-[a-zA-Z0-9]{6}$/);
  assert.equal(await realpath(root), root);
  assert.equal((await lstat(root)).mode & 0o777, 0o700);
  assert.notEqual(context.extensionMode, actual.ExtensionMode.Test, "Native dialogs require a normal development host");
  assert.equal(actual.workspace.isTrusted, true);
  assert.equal(actual.workspace.workspaceFolders?.length ?? 0, 0);
  const created = Date.now(), fixture = otherCancellationFixture(otherNativeCancelCases[0], created);
  // sourceWorkflowDraft timestamps itself; pin the fixture's observed creation
  // once before publication. Never rewrite its timestamps during the trial.
  fixture.record.updatedAt = new Date(created).toISOString();
  const storeRoot = join(root, "runner-v2"), store = new RunnerStore(storeRoot);
  await store.write(fixture.record);
  const initialSnapshot = await snapshot(storeRoot);
  const observation: DeploymentExpiryObservation = { createdAt: new Date(created).toISOString(), expiresAt: fixture.record.expiresAt,
    deployEntries: 0, modalEntries: 0, submitEntries: 0, storeWrites: 0, effectAttempts: [], controllerErrors: [], initialSnapshot, finalSnapshot: {} };
  const events: Record<string, unknown>[] = [], registry = new Map<string, () => unknown>();
  let receive: ((raw: unknown) => Promise<void>) | undefined, opened = false, complete = false;
  let readySettled!: () => void;
  const ready = new Promise<void>(resolve => { readySettled = resolve; });
  const event = (type: string, details: Record<string, unknown> = {}) => events.push({ type, ...details, at: new Date().toISOString() });
  const deny = (kind: string): never => { observation.effectAttempts.push(kind); event("denied-effect", { kind }); throw Error(`NO CLOUD fixture denied ${kind}`); };
  const finalize = async () => {
    if (complete) return;
    complete = true; observation.finalSnapshot = await snapshot(storeRoot);
    const failures = deploymentExpiryFailures(observation);
    await retain(join(root, "result.json"), { pass: failures.length === 0, failures, observation, events, hostPID: process.pid,
      evidenceLayer: "isolated-normal-native-production-controller-with-inert-effects", cloudRequests: 0,
      signedInCloudQualification: false, duplicateClickQualified: false, twoWindowQualified: false });
    await actual.window.showInformationMessage(failures.length ? "Expiry fixture did not qualify. Retained result.json explains why; no cloud action was available." : "Original preview expiry refusal observed; no submission or workflow change. Isolated native evidence only.");
  };
  const windowFacade = lazyNativeFacade(actual.window, {
    createWebviewPanel: (...args: Parameters<typeof actual.window.createWebviewPanel>) => {
      assert.equal(args[0], "agefreighter.runnerMigration");
      const panel = actual.window.createWebviewPanel(args[0], "NO CLOUD FIXTURE — " + args[1], args[2], args[3]);
      const webview = lazyNativeFacade(panel.webview, {
        postMessage: async (message: Record<string, unknown>) => {
          if (message.kind === "error") { observation.controllerErrors.push(String(message.text)); event("controller-refusal", { text: message.text }); }
          return panel.webview.postMessage(message);
        },
        onDidReceiveMessage: (listener, thisArgs, disposables) => {
          receive = async raw => {
            const action = (raw as { action?: string }).action;
            event("production-handler-entry", { action });
            if (!["ready", "accounts", "restore", "deploy"].includes(String(action))) return deny("unplanned panel action");
            if (action === "deploy") observation.deployEntries++;
            await listener.call(thisArgs, raw);
            if (action === "ready") readySettled();
            if (action === "deploy") await finalize();
          };
          return panel.webview.onDidReceiveMessage(receive, undefined, disposables);
        }
      });
      // Production assigns webview.html after creation. Forward setters too;
      // the namespace-only lazy facade otherwise writes its empty proxy target.
      return lazyNativeFacade(panel, { webview: new Proxy(webview, {
        set: (_target, key, value) => Reflect.set(panel.webview, key, value)
      }) });
    },
    showQuickPick: (async (items: { record?: { id: string } }[]) => {
      const selected = items.find(item => item.record?.id === fixture.record.id);
      assert.ok(selected); event("synthetic-restore-selection", { workflow: fixture.record.id }); return selected;
    }) as unknown as typeof actual.window.showQuickPick,
    showWarningMessage: (async (title: string, options: VSCode.MessageOptions, ...items: string[]) => {
      assert.equal(title, fixture.title); assert.equal(options.modal, true); assert.deepEqual(items, ["Create reviewed runner"]);
      observation.modalEntries++; observation.modalOpenedAt = new Date().toISOString(); event("actual-native-modal-open");
      await retain(join(root, "modal-opened.json"), { observation, title, options, items, storeSnapshot: await snapshot(storeRoot) });
      const choice = await actual.window.showWarningMessage(title, options, ...items);
      observation.modalReturnedAt = new Date().toISOString(); observation.nativeChoice = choice; event("actual-native-modal-return", { choice: choice ?? null });
      return choice;
    }) as typeof actual.window.showWarningMessage
  });
  const facade = lazyNativeFacade(actual, { window: windowFacade, commands: lazyNativeFacade(actual.commands, {
    registerCommand: (name, callback) => { registry.set(name, callback); return new actual.Disposable(() => registry.delete(name)); }
  }) });
  globalThis.__afDeploymentExpiry = {
    vscode: facade,
    azure: new Proxy({ subscriptions: async () => { event("synthetic-subscriptions"); return []; }, dispose: () => {} },
      { get: (target, key) => key in target ? Reflect.get(target, key) : () => deny(`Azure.${String(key)}`) }),
    denyFetch: () => deny("network fetch"),
    submitEntered: () => { observation.submitEntries++; event("submitRunner-entry"); },
    storeWrite: () => { observation.storeWrites++; deny("workflow persistence after setup"); }
  };
  const productionContext = { ...context, subscriptions: [] as VSCode.Disposable[], globalStorageUri: actual.Uri.file(root),
    secrets: new Proxy({}, { get: () => () => deny("credential access") }), extension: { packageJSON: { version: "2.4.0" } } } as unknown as VSCode.ExtensionContext;
  registerRunnerMigration(productionContext, { info: () => {}, error: () => {} } as unknown as VSCode.LogOutputChannel);
  context.subscriptions.push(...productionContext.subscriptions);
  context.subscriptions.push(actual.commands.registerCommand("agefreighterFixture.openDeploymentExpiry", async () => {
    if (opened) { await actual.window.showInformationMessage("This single-use preview is already open; its original expiry is unchanged."); return; }
    opened = true; registry.get("agefreighter.newGuidedMigration")!(); assert.ok(receive);
    // Observe completion of the real ready handler; a fixed delay could race
    // its busy guard and silently drop the synthetic restore prerequisite.
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try { await Promise.race([ready, new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(Error("Real webview ready handler did not settle")), 10_000); })]); }
    finally { if (timeout) clearTimeout(timeout); }
    await receive({ action: "restore" });
  }));
  await retain(join(root, "baseline.json"), { observation, fixtureSHA256: hash(JSON.stringify(fixture.record)), hostPID: process.pid,
    source: "A01 synthetic production-template preview; native deployment modal is real", originalPreview: fixture.record });
  const notice = await actual.workspace.openTextDocument({ language: "markdown", content:
    `# NO CLOUD — original deployment expiry refusal\n\nCreated ${observation.createdAt}\n\nOriginal expiry ${observation.expiresAt}\n\nRun **ISOLATED: Open deployment expiry preview**, then use the production panel's two approval checkboxes and Create button BEFORE expiry. Keep its actual modal open until AFTER the expiry above, then select **Create reviewed runner**. No cloud adapter exists; early confirmation fails this fixture. Do not change clocks or stored timestamps.\n\nThis single case does not test duplicate clicks, two windows, Azure response loss or guest bootstrap.\n\nEvidence: ${root}\n` });
  await actual.window.showTextDocument(notice, { preview: false });
}
