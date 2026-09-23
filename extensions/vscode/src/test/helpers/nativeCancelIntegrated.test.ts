import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { Script } from "node:vm";
import { integratedNativeCancelCases } from "./nativeCancelCatalog";

// Run the compiled integration up to presentation, then THROW. There is no GUI,
// no returned Cancel/approval, and no native/cloud cancellation credit. This
// deliberately produces failed trial ledgers and verifies they remain no-pass.
const extensionRoot = resolve(__dirname, "../../.."), harness = join(extensionRoot, "out-test/test/suite/nativeCancelHarness.js");
const boundary = "INTEGRATED_PRESENTATION_BOUNDARY_NOT_CANCEL";
for (const scenario of integratedNativeCancelCases) test(`${scenario.id} compiled integration reaches exactly one warning and records no false cancellation PASS`, async () => {
  const root = await mkdtemp("/private/tmp/af-native-cancel-"), previousRoot = process.env.AF_NATIVE_CANCEL_ROOT, previousCases = process.env.AF_NATIVE_CANCEL_CASES;
  process.env.AF_NATIVE_CANCEL_ROOT = root; process.env.AF_NATIVE_CANCEL_CASES = scenario.id;
  let warnings = 0;
  try {
    const vscode = { __esModule: true, ExtensionMode: { Development: 2 }, env: { appHost: "desktop" },
      Uri: { file: (path: string) => ({ fsPath: path, scheme: "file" }) }, ViewColumn: { One: 1, Beside: 2 },
      extensions: { getExtension: () => ({ extensionPath: extensionRoot }) }, commands: {},
      workspace: { isTrusted: true, workspaceFolders: undefined, openTextDocument: async () => ({}),
        getConfiguration: () => ({ inspect: () => ({ globalValue: true }) }) },
      window: { showTextDocument: async () => {}, showQuickPick: async (items: string[]) => items[0],
        showWarningMessage: async () => { warnings++; throw Error(boundary); } } };
    const module = { exports: {} as { run: (mode: number) => Promise<void> } }, native = createRequire(harness);
    const code = readFileSync(harness, "utf8");
    new Script(`(function(require,module,exports,__filename,__dirname){${code}\n})`, { filename: harness }).runInThisContext()(
      (name: string) => name === "vscode" ? vscode : native(name), module, module.exports, harness, dirname(harness));
    await assert.rejects(module.exports.run(2));
    assert.equal(warnings, 1);
    const ledger = JSON.parse(await readFile(join(root, "ledger.json"), "utf8"));
    assert.equal(ledger.pass, false); assert.equal(ledger.outcome, "failed"); assert.deepEqual(ledger.completed, []);
    assert.deepEqual(ledger.selected, [scenario.id]); assert.deepEqual(ledger.unexecuted, [scenario.id]);
    const names = (await readdir(root)).filter(name => name.startsWith(scenario.id + "-")); assert.equal(names.length, 1);
    const result = JSON.parse(await readFile(join(root, names[0]!, "result.json"), "utf8"));
    assert.equal(result.pass, false); assert.equal(result.actualNativeCancel, false); assert.equal(result.modalCount, 1);
    assert.deepEqual(result.deniedEffects, []); assert.equal(result.cloudRequests, 0); assert.equal(result.signedInCloudEvidence, false);
    assert.equal(result.preparationWrites, ["A14","A15","A17"].includes(scenario.id) ? 1 : 0);
    assert.deepEqual(result.after, result.modalBefore);
    assert.equal(result.events.filter((event: { type: string }) => event.type === "native-modal-result").length, 0);
  } finally {
    if (previousRoot === undefined) delete process.env.AF_NATIVE_CANCEL_ROOT; else process.env.AF_NATIVE_CANCEL_ROOT = previousRoot;
    if (previousCases === undefined) delete process.env.AF_NATIVE_CANCEL_CASES; else process.env.AF_NATIVE_CANCEL_CASES = previousCases;
    await rm(root, { recursive: true, force: true });
  }
});
