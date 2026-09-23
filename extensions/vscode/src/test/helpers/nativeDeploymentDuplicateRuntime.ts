/** Disposable native controller fixture. Never substitutes actual clicks or
 * window identities; all cloud transport is absent and explicitly inert. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, open, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import type * as VSCode from "vscode";
import { registerRunnerMigration } from "../../runnerMigration";
import { RunnerRecord } from "../../core/runner";
import { RunnerStore } from "../../guided/runnerStore";
import { otherCancellationFixture, otherNativeCancelCases } from "./nativeCancelOtherScenarios";
import { lazyNativeFacade } from "./nativeCancelFacade";
import { assertOnlySubmittedIntent, DuplicateMode, DuplicateObservation, duplicateObservationFailures, duplicateReadFixture } from "./nativeDeploymentDuplicate";

interface DuplicateBridge { vscode: typeof VSCode; azure: object; denyFetch: () => never; submitEntered: () => void; beforeWrite: (r: RunnerRecord) => Promise<void>; afterWrite: (r: RunnerRecord) => Promise<void> }
declare global { var __afDeploymentDuplicate: DuplicateBridge; }
const actual = module.require("vscode") as typeof VSCode;
const sha = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
async function retain(path: string, value: unknown) {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(JSON.stringify(value, null, 2) + "\n"); await file.sync(); } finally { await file.close(); }
}
async function json(path: string) { return JSON.parse(await readFile(path, "utf8")); }

export async function activate(context: VSCode.ExtensionContext) {
  const root = context.extensionPath;
  assert.match(root, /^\/private\/tmp\/af-deployment-duplicate-[a-zA-Z0-9]{6}$/);
  assert.equal(await realpath(root), root); assert.equal((await lstat(root)).mode & 0o777, 0o700);
  assert.notEqual(context.extensionMode, actual.ExtensionMode.Test); assert.equal(actual.workspace.isTrusted, true); assert.equal(actual.workspace.workspaceFolders?.length ?? 0, 0);
  const config = await json(join(root, "config.json")) as { mode: DuplicateMode; profile: string; storeRoot: string };
  assert.ok(config.mode === "same-window" || config.mode === "two-window");
  assert.equal(config.profile, join(root, "user-data")); assert.equal(config.storeRoot, join(root, "runner-v2"));
  // Actual host-provided storage location proves the shared profile; config
  // alone is not evidence that two launches actually used the same profile.
  assert.ok(context.globalStorageUri.fsPath.startsWith(config.profile + "/"));
  const identity = { hostPID: process.pid, sessionId: actual.env.sessionId, hostStorageUri: context.globalStorageUri.fsPath, activatedAt: new Date().toISOString() };
  assert.ok(identity.sessionId);
  let role: "A" | "B" = "A";
  try { await retain(join(root, "host-A.json"), identity); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    assert.equal(config.mode, "two-window", "Same-window fixture permits only one actual host");
    assert.notEqual((await json(join(root, "host-A.json"))).hostPID, process.pid);
    role = "B"; await retain(join(root, "host-B.json"), identity);
  }
  const store = new RunnerStore(config.storeRoot);
  if (role === "A") {
    const created = Date.now(), fixture = otherCancellationFixture(otherNativeCancelCases[0], created);
    fixture.record.updatedAt = new Date(created).toISOString();
    await store.write(fixture.record);
    await retain(join(root, "original.json"), fixture.record);
  }
  const original = await json(join(root, "original.json")) as RunnerRecord;
  const recordPath = join(config.storeRoot, original.id + ".json"), recordHash = async () => sha(await readFile(recordPath));
  assert.equal(await realpath(config.storeRoot), config.storeRoot);
  assert.equal(sha(JSON.stringify(await store.read(original.id))), sha(JSON.stringify(original)), "Both hosts must restore the same original preview before it advances");
  const observation: DuplicateObservation = { role, mode: config.mode, hostPID: process.pid, sessionId: identity.sessionId, profile: await realpath(config.profile), storeRoot: await realpath(config.storeRoot),
    deployEntries: 0, modalEntries: 0, submitEntries: 0, storeWrites: 0, inertPUTs: 0, errors: [], denied: [], initialSHA256: await recordHash(), expiresAt: original.expiresAt };
  const events: Record<string, unknown>[] = [], registry = new Map<string, () => unknown>();
  const event = (type: string, details: Record<string, unknown> = {}) => events.push({ type, ...details, at: new Date().toISOString() });
  const deny = (kind: string): never => { observation.denied.push(kind); event("denied-effect", { kind }); throw Error(`NO CLOUD duplicate fixture denied ${kind}`); };
  let receive: ((raw: unknown) => Promise<void>) | undefined, opened = false, done = false, readySettled!: () => void;
  let invokingDeployEntry = 0, acceptedDeployEntry = 0, modalOwnerEntry = 0;
  const ready = new Promise<void>(resolve => { readySettled = resolve; });
  const finish = async () => {
    if (done) return; done = true; observation.finalSHA256 = await recordHash();
    await retain(join(root, `result-${role}.json`), { observation, events, finalRecord: await store.read(original.id), host: identity,
      evidenceLayer: "isolated-normal-native-production-controller-with-inert-effects", cloudRequests: 0, lostAzureReplyQualified: false, requiresIndependentNativeInteractionObservation: true });
    if (role === "B" || config.mode === "same-window") {
      const a = role === "A" ? observation : (await json(join(root, "result-A.json"))).observation as DuplicateObservation;
      const failures = duplicateObservationFailures(a, role === "B" ? observation : undefined);
      await retain(join(root, "controller-result.json"), { controllerBoundaryPass: failures.length === 0, failures,
        mode: config.mode, actualNativeInteractionQualifiedByThisFile: false, operatorMustRetainActualWindowAndRepeatedActivationObservation: true,
        evidenceLayer: "isolated-normal-native-production-controller-with-inert-effects", signedInAzureDuplicationQualified: false });
    }
    await actual.window.showInformationMessage(`NO CLOUD host ${role} observation retained. Native interaction evidence must be independently reviewed; no Azure qualification is claimed.`);
  };
  const windowFacade = lazyNativeFacade(actual.window, {
    createWebviewPanel: (...args: Parameters<typeof actual.window.createWebviewPanel>) => {
      assert.equal(args[0], "agefreighter.runnerMigration");
      const panel = actual.window.createWebviewPanel(args[0], `NO CLOUD ${role} — ${args[1]}`, args[2], args[3]);
      const webview = lazyNativeFacade(panel.webview, {
        postMessage: async (message: Record<string, unknown>) => {
          if (message.kind === "error") { observation.errors.push(String(message.text)); event("controller-refusal", { text: message.text }); }
          if (message.kind === "busy") {
            // The accepted handler posts busy synchronously before its first
            // await. A concurrent rejected entry never owns that transition.
            if (message.value === true) acceptedDeployEntry = invokingDeployEntry;
            event("production-busy", { value: message.value, acceptedDeployEntry });
          }
          return panel.webview.postMessage(message);
        },
        onDidReceiveMessage: (listener, thisArgs, disposables) => {
          receive = async raw => {
            const action = (raw as { action?: string }).action; event("production-handler-entry", { action });
            if (!["ready", "restore", "deploy"].includes(String(action))) return deny("unplanned panel action");
            const entry = action === "deploy" ? ++observation.deployEntries : 0;
            invokingDeployEntry = entry;
            await listener.call(thisArgs, raw);
            if (action === "ready") readySettled();
            // A duplicate entry may return immediately while the first is
            // waiting on its native modal. Only the completed modal seals it.
            if (action === "deploy" && entry === modalOwnerEntry && observation.modalReturnedAt) await finish();
          };
          return panel.webview.onDidReceiveMessage(receive, undefined, disposables);
        }
      });
      return lazyNativeFacade(panel, { webview: new Proxy(webview, { set: (_target, key, value) => Reflect.set(panel.webview, key, value) }) });
    },
    showQuickPick: (async (items: { record?: { id: string } }[]) => { const selected = items.find(item => item.record?.id === original.id); assert.ok(selected); event("synthetic-original-restore"); return selected; }) as unknown as typeof actual.window.showQuickPick,
    showWarningMessage: (async (title: string, options: VSCode.MessageOptions, ...items: string[]) => {
      assert.equal(title, `Create the reviewed Linux discovery/migration VM ${original.vmId}?`); assert.equal(options.modal, true); assert.deepEqual(items, ["Create reviewed runner"]);
      modalOwnerEntry = acceptedDeployEntry;
      observation.modalEntries++; observation.modalOpenedAt = new Date().toISOString(); observation.modalSHA256 = await recordHash();
      assert.equal(observation.modalSHA256, observation.initialSHA256, "Original shared preview already advanced; start a fresh fixture");
      await retain(join(root, `modal-${role}.json`), { observation, title, options, items });
      const choice = await actual.window.showWarningMessage(title, options, ...items);
      observation.modalReturnedAt = new Date().toISOString(); observation.nativeChoice = choice; event("actual-native-modal-return", { choice: choice ?? null });
      return choice;
    }) as typeof actual.window.showWarningMessage
  });
  const reads = duplicateReadFixture(original);
  const azure = {
    dispose: () => {}, subscriptions: async () => { event("synthetic-subscriptions"); return []; },
    runnerList: async (subscription: string, path: string) => {
      assert.equal(subscription, original.input.subscriptionId); const result = reads.lists.get(path); if (!result) return deny("unplanned list"); event("synthetic-list", { path }); return result;
    },
    runnerRequest: async (subscription: string, path: string, method = "GET", body?: unknown) => {
      assert.equal(subscription, original.input.subscriptionId);
      if (method === "GET") { const result = reads.requests.get(path); if (!result) return deny("unplanned GET"); event("synthetic-GET", { path }); return result; }
      if (method === "POST" && path === `${original.deploymentId}/whatIf?api-version=2022-09-01`) {
        assert.deepEqual(body, { properties: { mode: "Incremental", template: original.template, whatIfSettings: { resultFormat: "ResourceIdOnly" } } }); event("synthetic-what-if"); return { status: 200, value: reads.whatIf };
      }
      if (method !== "PUT" || path !== `${original.deploymentId}?api-version=2022-09-01` || role !== "A") return deny("unplanned mutation");
      assert.deepEqual(body, { properties: { mode: "Incremental", template: original.template } });
      assertOnlySubmittedIntent(original, await store.read(original.id));
      if (config.mode === "two-window") { const b = (await json(join(root, "modal-B.json"))).observation; assert.equal(b.initialSHA256, observation.initialSHA256); assert.equal(b.modalSHA256, observation.initialSHA256); }
      await retain(join(root, "inert-put.json"), { role, host: identity, at: new Date().toISOString(), path, method, bodySHA256: sha(JSON.stringify(body)), intentSHA256: await recordHash(), syntheticTransport: true });
      observation.inertPUTs++; event("inert-PUT-counted-no-network"); return { status: 202, value: {} };
    }
  };
  globalThis.__afDeploymentDuplicate = {
    vscode: lazyNativeFacade(actual, { window: windowFacade, commands: lazyNativeFacade(actual.commands, { registerCommand: (name, callback) => { registry.set(name, callback); return new actual.Disposable(() => registry.delete(name)); } }) }),
    azure: new Proxy(azure, { get: (target, key) => key in target ? Reflect.get(target, key) : () => deny(`Azure.${String(key)}`) }), denyFetch: () => deny("network fetch"),
    submitEntered: () => { observation.submitEntries++; event("submitRunner-entry"); },
    beforeWrite: async record => {
      observation.storeWrites++; if (role !== "A" || observation.storeWrites !== 1) return deny("extra workflow write");
      assertOnlySubmittedIntent(original, record); assert.equal(await recordHash(), observation.initialSHA256);
      await retain(join(root, "intent-authorized.json"), { original, intended: record, observation, at: new Date().toISOString() });
    },
    afterWrite: async record => { await retain(join(root, "intent-persisted.json"), { record, sha256: await recordHash(), at: new Date().toISOString() }); }
  };
  const productionContext = { ...context, subscriptions: [] as VSCode.Disposable[], globalStorageUri: actual.Uri.file(root),
    secrets: new Proxy({}, { get: () => () => deny("credential access") }), extension: { packageJSON: { version: "2.4.0" } } } as unknown as VSCode.ExtensionContext;
  registerRunnerMigration(productionContext, { info: () => {}, error: () => {} } as unknown as VSCode.LogOutputChannel); context.subscriptions.push(...productionContext.subscriptions);
  context.subscriptions.push(actual.commands.registerCommand("agefreighterFixture.openDeploymentDuplicate", async () => {
    if (opened) return; opened = true; registry.get("agefreighter.newGuidedMigration")!(); assert.ok(receive);
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try { await Promise.race([ready, new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(Error("Real webview ready did not settle")), 10_000); })]); }
    finally { if (timeout) clearTimeout(timeout); }
    await receive({ action: "restore" });
  }));
  await retain(join(root, `baseline-${role}.json`), { observation, host: identity, original, fixtureSynthetic: true });
  const notice = await actual.workspace.openTextDocument({ language: "markdown", content: `# NO CLOUD — host ${role}, ${config.mode}\n\nOriginal preview expires ${original.expiresAt}.\n\nRun **ISOLATED: Open deployment duplicate preview**. Both hosts must restore the original BEFORE any approval. For two windows: open B's actual modal first; leave it pending, then approve A, wait for result-A.json, finally approve B. Use the identical disposable profile for both actual windows. For same-window: independently observe actual rapid repeated activation and busy/disabled UI; this harness never injects a click or duplicate message.\n\nAt most one inert PUT is available; no network transport or credentials. Snapshot/intent differences are retained separately. Native interaction must be independently observed; no Azure duplicate/lost-reply/bootstrap qualification.\n\nEvidence ${root}\n` });
  await actual.window.showTextDocument(notice, { preview: false });
}
