import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import test from "node:test";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import { deploymentExpiryFailures, DeploymentExpiryObservation } from "./nativeDeploymentExpiry";
import { prepareDeploymentExpiryCompanion } from "./nativeDeploymentExpiryCompanion";

function observation(): DeploymentExpiryObservation {
  return { createdAt: "2026-09-23T00:00:00.000Z", expiresAt: "2026-09-23T00:15:00.000Z", modalOpenedAt: "2026-09-23T00:01:00.000Z",
    modalReturnedAt: "2026-09-23T00:15:01.000Z", nativeChoice: "Create reviewed runner", deployEntries: 1, modalEntries: 1, submitEntries: 0, storeWrites: 0,
    effectAttempts: [], controllerErrors: ["The approved preview changed or expired. Reconnect and review deployment again."],
    initialSnapshot: { "record.json": "384:original-sha" }, finalSnapshot: { "record.json": "384:original-sha" } };
}
test("expiry evidence evaluator accepts only the original fifteen-minute native refusal observation", () => {
  assert.deepEqual(deploymentExpiryFailures(observation()), []);
});
const rejected: [string, (o: DeploymentExpiryObservation) => void][] = [
  ["early approval", o => { o.modalReturnedAt = "2026-09-23T00:14:59.999Z"; }],
  ["already expired before modal", o => { o.modalOpenedAt = o.expiresAt; }],
  ["shortened original expiry", o => { o.expiresAt = "2026-09-23T00:14:00.000Z"; }],
  ["missing timestamp", o => { delete o.modalReturnedAt; }],
  ["Cancel", o => { delete o.nativeChoice; }],
  ["duplicate handler entry", o => { o.deployEntries++; }],
  ["duplicate modal entry", o => { o.modalEntries++; }],
  ["submit entry", o => { o.submitEntries++; }],
  ["workflow persistence", o => { o.storeWrites++; }],
  ["denied network attempt", o => { o.effectAttempts.push("fetch"); }],
  ["wrong refusal", o => { o.controllerErrors = ["Network failed"]; }],
  ["changed private store", o => { o.finalSnapshot["record.json"] = "384:changed-sha"; }]
];
for (const [name, change] of rejected) test(`expiry evidence rejects ${name}`, () => {
  const value = observation(); change(value); assert.ok(deploymentExpiryFailures(value).length > 0);
});

test("expiry preparation is isolated, hash-pinned, no GUI or clock change, and excludes the account adapter", async () => {
  const root = await mkdtemp("/private/tmp/af-deployment-expiry-");
  try {
    const extensionRoot = resolve(__dirname, "../../.."), args = await prepareDeploymentExpiryCompanion(root, extensionRoot);
    assert.deepEqual(args.filter(arg => arg.startsWith("--extensionDevelopmentPath=")), [`--extensionDevelopmentPath=${root}`]);
    assert.ok(!args.some(arg => /extensionTests|install-extension|disable-workspace-trust/.test(arg)));
    const manifest = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
    assert.deepEqual(manifest.contributes.commands, [{ command: "agefreighterFixture.openDeploymentExpiry", title: "ISOLATED: Open deployment expiry preview" }]);
    const preparation = JSON.parse(await readFile(join(root, "preparation.json"), "utf8"));
    const bytes = await readFile(join(root, "runtime.cjs")), text = bytes.toString();
    assert.equal(preparation.bundleSHA256, createHash("sha256").update(bytes).digest("hex"));
    assert.equal(preparation.expiryStartsAtActivation, true);
    assert.equal(preparation.signedInCloudQualification, false);
    assert.ok(!preparation.inputs.some((p: string) => /(?:^|\/)guided\/azure\.ts$/.test(p)));
    for (const path of ["expiry-fixture:vscode", "expiry-fixture:azure", "expiry-fixture:lifecycle", "expiry-fixture:store"]) assert.ok(preparation.inputs.includes(path), path);
    assert.match(text, /module\.require\("vscode"\)/);
    assert.match(text, /__afDeploymentExpiry\.submitEntered\(\)/);
    assert.match(text, /__afDeploymentExpiry\.storeWrite\(\)/);
    assert.doesNotMatch(text, /Date\.now\s*=|setSystemTime|useFakeTimers/);
    await assert.rejects(readFile(join(root, "baseline.json")), /ENOENT/);
    await assert.rejects(prepareDeploymentExpiryCompanion(root, extensionRoot), /EEXIST/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("expiry preparation rejects arbitrary operator paths before writing", async () => {
  await assert.rejects(prepareDeploymentExpiryCompanion("/not-an-isolated-fixture", "unused"));
});

test("bundled production panel skips enumerable proposed context getters and reaches inert mock dialog; mock Cancel never qualifies expiry", async () => {
  const root = await mkdtemp("/private/tmp/af-deployment-expiry-");
  try {
    await prepareDeploymentExpiryCompanion(root, resolve(__dirname, "../../.."));
    const commands = new Map<string, () => Promise<void>>();
    let listener: ((raw: unknown) => Promise<void>) | undefined, rendered = "", warnings = 0;
    const messages: Record<string, unknown>[] = [];
    class Disposable { constructor(readonly dispose: () => void) {} }
    const fake = {
      ExtensionMode: { Test: 3 }, ViewColumn: { One: 1 }, Disposable,
      Uri: { file: (fsPath: string) => ({ fsPath }) },
      workspace: { isTrusted: true, workspaceFolders: [], openTextDocument: async () => ({}) },
      commands: { registerCommand: (name: string, callback: () => Promise<void>) => { commands.set(name, callback); return new Disposable(() => {}); } },
      window: {
        createWebviewPanel: () => ({ onDidDispose: () => new Disposable(() => {}), webview: {
          cspSource: "fixture", set html(value: string) { rendered = value; },
          postMessage: async (message: Record<string, unknown>) => { messages.push(message); return true; },
          onDidReceiveMessage: (callback: (raw: unknown) => Promise<void>) => { listener = callback; queueMicrotask(() => void callback({ action: "ready" })); return new Disposable(() => {}); }
        } }),
        showWarningMessage: async () => { warnings++; return undefined; },
        showInformationMessage: async () => undefined, showTextDocument: async () => undefined
      }
    };
    const realRequire = createRequire(__filename), compiled = { exports: {} as { activate: (context: unknown) => Promise<void> } };
    const customRequire = (name: string) => name === "vscode" ? fake : realRequire(name);
    runInNewContext(await readFile(join(root, "runtime.cjs"), "utf8"), { module: Object.assign(compiled, { require: customRequire }), exports: compiled.exports,
      require: customRequire, process, Buffer, URL, setTimeout, clearTimeout, console });
    let proposedGetterReads = 0;
    const context = { extensionPath: root, extensionMode: 2, subscriptions: [] };
    Object.defineProperty(context, "extensionRuntime", { enumerable: true, get: () => { proposedGetterReads++; throw Error("Proposed extensionRuntime API is unavailable"); } });
    await compiled.exports.activate(context);
    assert.equal(proposedGetterReads, 0);
    assert.deepEqual([...commands.keys()], ["agefreighterFixture.openDeploymentExpiry"]);
    await commands.get("agefreighterFixture.openDeploymentExpiry")!();
    assert.match(rendered, /Approve & deploy discovery VM/);
    const record = messages.find(m => m.kind === "record")!.record as { id: string; previewHash: string };
    await listener!({ action: "deploy", workflow: record.id, hash: record.previewHash, networkApproved: true, costApproved: true });
    assert.equal(warnings, 1);
    const result = JSON.parse(await readFile(join(root, "result.json"), "utf8"));
    assert.equal(result.pass, false); assert.equal(result.observation.deployEntries, 1); assert.equal(result.observation.modalEntries, 1);
    assert.equal(result.observation.submitEntries, 0); assert.equal(result.observation.storeWrites, 0);
    assert.deepEqual(result.observation.effectAttempts, []);
    assert.deepEqual(result.observation.initialSnapshot, result.observation.finalSnapshot);
    assert.equal(proposedGetterReads, 0);
  } finally { await rm(root, { recursive: true, force: true }); }
});
