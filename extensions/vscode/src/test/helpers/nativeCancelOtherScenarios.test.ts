import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { mkdtemp, mkdir, readFile, readdir, open, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { RunnerRecord } from "../../core/runner";
import { RunnerStore } from "../../guided/runnerStore";
import { otherCancellationFixture, otherNativeCancelCases } from "./nativeCancelOtherScenarios";

// Presentation-boundary tests, not cancellation tests: throw before a warning
// can return any selection. Never mock Cancel, positive consent or cloud writes.
const boundary = "INERT_MODAL_PRESENTATION_BOUNDARY";
async function snapshot(root: string) {
  const result: Record<string, string> = {};
  for (const name of (await readdir(root)).sort()) result[name] = createHash("sha256").update(await readFile(join(root, name))).digest("hex");
  return result;
}
function withoutRevision(r: RunnerRecord) { const { updatedAt: _updatedAt, ...rest } = r; return rest; }

test("non-source expansion remains exactly seven frozen decision IDs", () => {
  assert.deepEqual(otherNativeCancelCases.map(s => s.id), ["A01", "A02", "A03", "A17", "A21", "A22", "A28"]);
});

for (const scenario of otherNativeCancelCases) test(`${scenario.id} actual production prerequisites reach its warning boundary without unapproved effects`, async () => {
  const root = await mkdtemp(join(tmpdir(), "af-other-cancel-prerequisite-"));
  try {
    const fixture = otherCancellationFixture(scenario), storeRoot = join(root, "runner-v2"), store = new RunnerStore(storeRoot);
    await store.write(fixture.record);
    for (const report of fixture.reports) await store.retainReport(fixture.record.id, report.manifest, report.text);
    await mkdir(join(root, "fixtures"), { mode: 0o700 });
    for (const file of fixture.files) {
      const h = await open(join(root, "fixtures", file.name), "wx", 0o600);
      try { await h.writeFile(file.text); } finally { await h.close(); }
    }
    const before = await snapshot(storeRoot), original = await store.read(fixture.record.id);
    const violations: string[] = [], warnings: string[] = [], reads: string[] = [], posts: Record<string, unknown>[] = [];
    let observations = 0, modalReached = false;
    const deny = (name: string): never => { violations.push(name); throw Error(`Denied ${name}`); };
    const control = {
      persist: async (record: RunnerRecord) => {
        assert.equal(modalReached, false); assert.equal(++observations, fixture.preModalObservationWrites);
        assert.deepEqual(withoutRevision(record), withoutRevision(original), "A17 may only reconcile the same failed target status");
        await store.write(record);
      },
      request: async (_subscription: string, path: string, method = "GET") => {
        if (method !== "GET") return deny(`ARM ${method}`);
        const key = path.split("?")[0]!;
        if (!fixture.responses.has(key)) return deny(`unexpected GET ${key}`);
        reads.push(key); return { status: 200, value: structuredClone(fixture.responses.get(key)) };
      },
      list: async (_subscription: string, path: string) => {
        const key = path.split("?")[0]!;
        if (!fixture.lists.has(key)) return deny(`unexpected list ${key}`);
        reads.push(key); return structuredClone(fixture.lists.get(key));
      }, sleep: async () => deny("wait")
    };
    const commands = new Map<string, () => unknown>();
    let receive = async (_message: unknown) => {};
    const vscode = { ViewColumn: { One: 1, Beside: 2 }, workspace: { isTrusted: true,
      getConfiguration: () => ({ inspect: () => ({ globalValue: fixture.requiresDevelopmentOptIn }) }) },
      commands: { registerCommand: (name: string, fn: () => unknown) => { commands.set(name, fn); return { dispose() {} }; } },
      window: {
        showQuickPick: async (items: unknown[]) => fixture.action ? items.find(x => x === fixture.action) : items[0],
        showOpenDialog: async () => {
          assert.ok(fixture.openFile); return [{ scheme: "file", fsPath: join(root, "fixtures", fixture.openFile) }];
        },
        showWarningMessage: async (title: string, options: { modal: boolean }) => {
          modalReached = true; warnings.push(title); assert.equal(options.modal, true); throw Error(boundary);
        },
        showInformationMessage: async () => deny("unexpected information"),
        showErrorMessage: async () => deny("unexpected native error"),
        createWebviewPanel: () => ({ onDidDispose() {}, reveal() {}, webview: { cspSource: "synthetic", html: "",
          onDidReceiveMessage: (fn: typeof receive) => { receive = fn; }, postMessage: async (message: Record<string, unknown>) => { posts.push(message); } } })
      } };
    const context = { subscriptions: [], extension: { packageJSON: { version: "2.4.0" } }, globalStorageUri: { fsPath: root },
      secrets: new Proxy({}, { get: (_t, name) => () => deny(`secret.${String(name)}`) }) };
    const azure = new Proxy({}, { get: (_t, name) => () => deny(`azure.${String(name)}`) });
    const cache = new Map<string, { exports: Record<string, (...args: any[]) => any> }>();
    const sourceRoot = join(__dirname, "../..");
    const load = (path: string): Record<string, (...args: any[]) => any> => {
      const prior = cache.get(path); if (prior) return prior.exports;
      const module = { exports: {} as Record<string, (...args: any[]) => any> }; cache.set(path, module);
      const native = createRequire(path), code = transformSync(readFileSync(path, "utf8"), { loader: "ts", format: "cjs" }).code;
      new Script(`(function(require,module,exports,__filename,__dirname){${code}\n})`, { filename: path }).runInThisContext()(
        (name: string) => {
          if (name === "vscode") return vscode;
          const resolved = native.resolve(name);
          if (resolved === join(sourceRoot, "guided/azure.ts")) return { AzureSession: class {
            runnerRequest = control.request; runnerList = control.list;
            subscriptions = () => deny("account enumeration");
          } };
          return name.startsWith(".") && resolved.startsWith(sourceRoot) && resolved.endsWith(".ts") ? load(resolved) : native(name);
        }, module, module.exports, path, dirname(path));
      return module.exports;
    };
    const handler = load(join(sourceRoot, `${scenario.module}.ts`))[scenario.exported]!;
    try {
      if (scenario.id === "A01") {
        handler(context, { info() {}, error() {} });
        assert.ok(commands.has("agefreighter.newGuidedMigration"));
        await commands.get("agefreighter.newGuidedMigration")!();
        for (const message of fixture.prepareMessages) await receive(message);
        await receive(fixture.message);
        assert.equal(posts.filter(x => x.kind === "error").at(-1)?.text, boundary);
      } else if (scenario.id === "A02" || scenario.id === "A03") await handler(control, store, azure);
      else await handler(context, control, store, azure, fixture.record.id);
    } catch (error) { assert.equal((error as Error).message, boundary); }
    assert.deepEqual(warnings, [fixture.title]); assert.deepEqual(violations, []);
    assert.equal(observations, fixture.preModalObservationWrites);
    if (observations) {
      assert.deepEqual(withoutRevision(await store.read(fixture.record.id)), withoutRevision(original));
      const after = await snapshot(storeRoot);
      for (const name of Object.keys(before).filter(name => name !== `${fixture.record.id}.json`)) assert.equal(after[name], before[name]);
      assert.deepEqual(Object.keys(after), Object.keys(before));
    } else assert.deepEqual(await snapshot(storeRoot), before);
    if (!["A17", "A21"].includes(scenario.id)) assert.deepEqual(reads, []);
  } finally { await rm(root, { recursive: true, force: true }); }
});
