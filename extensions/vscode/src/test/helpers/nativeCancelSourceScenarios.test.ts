import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { RunnerRecord } from "../../core/runner";
import { verifyReportBytes } from "../../core/runnerBlob";
import { RunnerStore } from "../../guided/runnerStore";
import { sourceCancellationFixture, sourceNativeCancelCases } from "./nativeCancelSourceScenarios";

// Prerequisite-contract checks only. The warning adapter throws at presentation;
// it never returns a synthetic Cancel or approval and never opens a native UI.
// Production core validation is loaded normally. No cloud/credential adapter is
// supplied. Reports and records exist only in a fresh disposable private store.
const panelPath = join(__dirname, "../../runnerSourcePanel.ts");
const code = transformSync(readFileSync(panelPath, "utf8"), { loader: "ts", format: "cjs" }).code;
const native = createRequire(panelPath);
async function snapshot(root: string) {
  const files: Record<string, string> = {};
  for (const name of (await readdir(root)).sort()) files[name] = createHash("sha256").update(await readFile(join(root, name))).digest("hex");
  return files;
}

test("source prerequisite IDs are exactly the ten frozen uncredited source decisions", () => {
  assert.deepEqual(sourceNativeCancelCases.map(x => x.id), ["A04", "A05", "A06", "A07", "A08", "A10", "A12", "A13", "A14", "A15"]);
  assert.equal(new Set(sourceNativeCancelCases.map(x => x.title)).size, 10);
});

for (const scenario of sourceNativeCancelCases) test(`${scenario.id} synthetic prerequisites reach the exact production modal without effects`, async () => {
  const root = await mkdtemp(join(tmpdir(), "af-source-cancel-prerequisite-"));
  try {
    const f = sourceCancellationFixture(scenario), store = new RunnerStore(root);
    await store.write(f.record);
    for (const report of f.reports) {
      verifyReportBytes(Buffer.from(report.text), report.manifest);
      await store.retainReport(f.record.id, report.manifest, report.text);
    }
    const violations: string[] = [], warnings: string[] = [], messages: Record<string, unknown>[] = [];
    let receive = async (_message: unknown) => {}, setup = true, setupWrites = 0;
    const deny = (name: string): never => { violations.push(name); throw Error(`Denied inert fixture effect: ${name}`); };
    const inertStore = {
      read: (id: string) => store.read(id), readReport: store.readReport.bind(store),
      exclusive: store.exclusive.bind(store),
      write: async (record: RunnerRecord) => {
        if (!setup || !f.baselineAfterPrepare) return deny("record write");
        assert.equal(record.id, f.record.id); setupWrites++; await store.write(record);
      }
    };
    const control = new Proxy({}, { get: (_target, name) => () => deny(`control.${String(name)}`) });
    const services = new Proxy({}, { get: (_target, name) => () => deny(`service.${String(name)}`) });
    const modules: Record<string, unknown> = {
      vscode: { workspace: { isTrusted: true }, ViewColumn: { One: 1 }, window: {
        createWebviewPanel: () => ({ onDidDispose: () => {}, webview: {
          html: "", postMessage: async (message: Record<string, unknown>) => { messages.push(message); },
          onDidReceiveMessage: (listener: typeof receive) => { receive = listener; return { dispose: () => {} }; }
        } }),
        showWarningMessage: async (title: string, options: { modal: boolean }) => {
          assert.equal(setup, false); assert.equal(options.modal, true);
          warnings.push(title); throw Error("INERT_MODAL_PRESENTATION_BOUNDARY");
        },
        showInputBox: () => deny("credential input"), showOpenDialog: () => deny("file selection"),
        withProgress: () => deny("unexpected progress/work")
      } },
      "./sourceCredentialPanel": new Proxy({}, { get: (_target, name) => () => deny(`credential.${String(name)}`) }),
      "./runnerWatch": { watchRetainedOperation: () => deny("background watcher") }
    };
    const module = { exports: {} as { openRunnerSource: (...args: unknown[]) => void } };
    new Script(code).runInNewContext({ module, exports: module.exports, Error, Buffer,
      require: (name: string) => name in modules ? modules[name] : native(name) });
    module.exports.openRunnerSource({ subscriptions: [], secrets: new Proxy({}, { get: () => () => deny("secret access") }) },
      control, inertStore, f.record.id, f.requiresServices ? services : undefined);
    for (const message of f.prepareMessages) await receive(message);
    assert.deepEqual(warnings, []); assert.deepEqual(violations, []);
    assert.equal(messages.some(message => message.kind === "error"), false, JSON.stringify(messages));
    assert.equal(setupWrites, f.prepareMessages.length);
    const before = await snapshot(root);
    setup = false;
    await receive(f.message);
    assert.deepEqual(warnings, [scenario.title]);
    assert.deepEqual(violations, []);
    assert.deepEqual(await snapshot(root), before);
    assert.equal(messages.filter(message => message.kind === "error").at(-1)?.text, "INERT_MODAL_PRESENTATION_BOUNDARY");
    assert.equal(f.responses.size, 0); // No cloud-like reads are needed to reach these modals.
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("new-transfer prerequisites cannot silently reuse a prior transfer approval", () => {
  for (const id of ["A05", "A13"]) {
    const scenario = sourceNativeCancelCases.find(x => x.id === id)!;
    assert.equal(sourceCancellationFixture(scenario).record.reportTransfers, undefined);
  }
});

test("assessment-read prerequisites require real local review before the baseline", () => {
  for (const scenario of sourceNativeCancelCases) {
    const f = sourceCancellationFixture(scenario), review = scenario.id === "A14" || scenario.id === "A15";
    assert.equal(f.baselineAfterPrepare, review);
    assert.deepEqual(f.prepareMessages.map(m => m.action), review ? ["review"] : []);
    if (review) assert.equal(f.record.sourceDraft, undefined);
  }
});
