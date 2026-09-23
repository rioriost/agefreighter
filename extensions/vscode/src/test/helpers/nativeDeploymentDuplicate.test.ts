import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import test from "node:test";
import { assertOnlySubmittedIntent, DuplicateObservation, duplicateObservationFailures } from "./nativeDeploymentDuplicate";
import { prepareDeploymentDuplicateCompanion } from "./nativeDeploymentDuplicateCompanion";
import { otherCancellationFixture, otherNativeCancelCases } from "./nativeCancelOtherScenarios";

function observations(): [DuplicateObservation, DuplicateObservation] {
  const a: DuplicateObservation = { role: "A", mode: "two-window", hostPID: 100, sessionId: "synthetic-A", profile: "/synthetic/profile", storeRoot: "/synthetic/store",
    deployEntries: 1, modalEntries: 1, submitEntries: 1, storeWrites: 1, inertPUTs: 1, nativeChoice: "Create reviewed runner", errors: [], denied: [],
    initialSHA256: "original", modalSHA256: "original", finalSHA256: "submitted", modalOpenedAt: "2026-09-23T00:00:01Z", modalReturnedAt: "2026-09-23T00:00:03Z", expiresAt: "2026-09-23T00:15:00Z" };
  return [a, { ...a, role: "B", hostPID: 101, sessionId: "synthetic-B", submitEntries: 0, storeWrites: 0, inertPUTs: 0,
    errors: ["The approved preview changed or expired. Reconnect and review deployment again."], modalOpenedAt: "2026-09-23T00:00:00Z", modalReturnedAt: "2026-09-23T00:00:04Z" }];
}
test("synthetic observation evaluator accepts distinct host/shared profile boundary but is not native evidence", () => {
  assert.deepEqual(duplicateObservationFailures(...observations()), []);
});
for (const [name, change] of [
  ["same host", (a: DuplicateObservation, b: DuplicateObservation) => { b.hostPID = a.hostPID; }],
  ["different profile", (_a: DuplicateObservation, b: DuplicateObservation) => { b.profile += "other"; }],
  ["different store", (_a: DuplicateObservation, b: DuplicateObservation) => { b.storeRoot += "other"; }],
  ["extra PUT", (_a: DuplicateObservation, b: DuplicateObservation) => { b.inertPUTs++; }],
  ["extra intent", (_a: DuplicateObservation, b: DuplicateObservation) => { b.storeWrites++; }],
  ["changed original review", (_a: DuplicateObservation, b: DuplicateObservation) => { b.modalSHA256 = "changed"; }],
  ["secondary review after primary", (_a: DuplicateObservation, b: DuplicateObservation) => { b.modalOpenedAt = "2026-09-23T00:00:05Z"; }],
  ["missing refusal", (_a: DuplicateObservation, b: DuplicateObservation) => { b.errors = []; }],
  ["expiry rather than duplicate guard", (_a: DuplicateObservation, b: DuplicateObservation) => { b.modalReturnedAt = b.expiresAt; }],
  ["denied effect", (a: DuplicateObservation) => { a.denied.push("network"); }]
] as const) test(`duplicate observation rejects ${name}`, () => { const [a, b] = observations(); change(a, b); assert.ok(duplicateObservationFailures(a, b).length); });

test("only phase and observed revision may change during the single legitimate intent", () => {
  const original = otherCancellationFixture(otherNativeCancelCases[0]).record;
  const submitted = { ...original, phase: "deployment-submitted" as const, updatedAt: new Date().toISOString() };
  assert.doesNotThrow(() => assertOnlySubmittedIntent(original, submitted));
  for (const value of [{ ...submitted, previewHash: "different" }, { ...submitted, expiresAt: "renewed" }, { ...submitted, hourlyComputeUSD: 900 }, { ...submitted, phase: "unknown" as const }]) assert.throws(() => assertOnlySubmittedIntent(original, value));
});

test("bundle skips enumerable proposed context getters; synthetic duplicate is dropped with one inert PUT and exact intent, no native claim", async () => {
  const root = await mkdtemp("/private/tmp/af-deployment-duplicate-");
  try {
    const args = await prepareDeploymentDuplicateCompanion(root, resolve(__dirname, "../../.."), "same-window");
    assert.deepEqual(args.filter(arg => arg.startsWith("--extensionDevelopmentPath=")), [`--extensionDevelopmentPath=${root}`]);
    assert.ok(!args.some(arg => /extensionTests|install-extension|disable-workspace-trust/.test(arg)));
    const commands = new Map<string, () => Promise<void>>(), messages: Record<string, unknown>[] = [];
    let listener: ((raw: unknown) => Promise<void>) | undefined, rendered = "", resolveModal!: (choice: string) => void, sawModal!: () => void;
    const modal = new Promise<string>(resolve => { resolveModal = resolve; }), modalSeen = new Promise<void>(resolve => { sawModal = resolve; });
    class Disposable { constructor(readonly dispose: () => void) {} }
    const fake = { ExtensionMode: { Test: 3 }, ViewColumn: { One: 1 }, Disposable, env: { sessionId: "explicitly-mocked-not-native" },
      Uri: { file: (fsPath: string) => ({ fsPath }) }, workspace: { isTrusted: true, workspaceFolders: [], openTextDocument: async () => ({}) },
      commands: { registerCommand: (name: string, callback: () => Promise<void>) => { commands.set(name, callback); return new Disposable(() => {}); } },
      window: { createWebviewPanel: () => ({ onDidDispose: () => new Disposable(() => {}), webview: {
        cspSource: "fixture", set html(value: string) { rendered = value; }, postMessage: async (message: Record<string, unknown>) => { messages.push(message); return true; },
        onDidReceiveMessage: (callback: (raw: unknown) => Promise<void>) => { listener = callback; queueMicrotask(() => void callback({ action: "ready" })); return new Disposable(() => {}); }
      } }), showWarningMessage: () => { sawModal(); return modal; }, showInformationMessage: async () => undefined, showTextDocument: async () => undefined }
    };
    const realRequire = createRequire(__filename), compiled = { exports: {} as { activate: (context: unknown) => Promise<void> } };
    const customRequire = (name: string) => name === "vscode" ? fake : realRequire(name);
    runInNewContext(await readFile(join(root, "runtime.cjs"), "utf8"), { module: Object.assign(compiled, { require: customRequire }), exports: compiled.exports,
      require: customRequire, process, Buffer, URL, setTimeout, clearTimeout, console });
    let proposedGetterReads = 0;
    const context = { extensionPath: root, extensionMode: 2, subscriptions: [], globalStorageUri: { fsPath: join(root, "user-data/User/globalStorage/fixture") } };
    Object.defineProperty(context, "extensionRuntime", { enumerable: true, get: () => { proposedGetterReads++; throw Error("Proposed extensionRuntime API is unavailable"); } });
    await compiled.exports.activate(context);
    assert.equal(proposedGetterReads, 0);
    assert.deepEqual([...commands.keys()], ["agefreighterFixture.openDeploymentDuplicate"]);
    await commands.get("agefreighterFixture.openDeploymentDuplicate")!(); assert.match(rendered, /Approve & deploy discovery VM/);
    const record = messages.find(m => m.kind === "record")!.record as { id: string; previewHash: string };
    const deploy = { action: "deploy", workflow: record.id, hash: record.previewHash, networkApproved: true, costApproved: true };
    const first = listener!(deploy); await modalSeen; await listener!(deploy);
    await assert.rejects(readFile(join(root, "result-A.json")), /ENOENT/);
    resolveModal("Create reviewed runner"); await first;
    const result = JSON.parse(await readFile(join(root, "result-A.json"), "utf8")), verdict = JSON.parse(await readFile(join(root, "controller-result.json"), "utf8"));
    assert.equal(result.observation.deployEntries, 2); assert.equal(result.observation.modalEntries, 1); assert.equal(result.observation.submitEntries, 1);
    assert.equal(result.observation.storeWrites, 1); assert.equal(result.observation.inertPUTs, 1); assert.deepEqual(result.observation.denied, []);
    assert.equal(result.finalRecord.phase, "deployment-submitted"); assert.equal(verdict.controllerBoundaryPass, true);
    assert.equal(verdict.actualNativeInteractionQualifiedByThisFile, false); assert.equal(verdict.signedInAzureDuplicationQualified, false);
    assert.equal(result.lostAzureReplyQualified, false);
    assert.equal(proposedGetterReads, 0);
    await assert.rejects(prepareDeploymentDuplicateCompanion(root, resolve(__dirname, "../../.."), "same-window"), /EEXIST/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
