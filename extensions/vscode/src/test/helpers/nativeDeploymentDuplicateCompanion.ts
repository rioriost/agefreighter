/** Preparation only. No launch/install, credentials or operator store access. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import { build } from "esbuild";
import { DuplicateMode } from "./nativeDeploymentDuplicate";

async function retain(path: string, bytes: string | Uint8Array) {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
}
export async function prepareDeploymentDuplicateCompanion(root: string, extensionRoot: string, mode: DuplicateMode): Promise<string[]> {
  assert.ok(mode === "same-window" || mode === "two-window");
  assert.match(root, /^\/private\/tmp\/af-deployment-duplicate-[a-zA-Z0-9]{6}$/);
  assert.equal(await realpath(root), root);
  const info = await lstat(root); assert.ok(info.isDirectory() && !info.isSymbolicLink()); assert.equal(info.mode & 0o777, 0o700);
  await retain(join(root, "config.json"), JSON.stringify({ mode, profile: join(root, "user-data"), storeRoot: join(root, "runner-v2") }));
  const source = join(extensionRoot, "src"), lifecycle = join(source, "core/runnerLifecycle.ts"), store = join(source, "guided/runnerStore.ts");
  const result = await build({ entryPoints: [join(source, "test/helpers/nativeDeploymentDuplicateRuntime.ts")], bundle: true, platform: "node", format: "cjs", target: "node22",
    write: false, metafile: true, logLevel: "silent", external: ["vscode"], define: { fetch: "globalThis.__afDeploymentDuplicate.denyFetch" },
    plugins: [{ name: "duplicate-inert-boundaries", setup(builder) {
      builder.onResolve({ filter: /^vscode$/ }, args => args.kind === "import-statement" ? { path: "vscode", namespace: "duplicate-fixture" } : undefined);
      builder.onResolve({ filter: /\/guided\/azure$/ }, () => ({ path: "azure", namespace: "duplicate-fixture" }));
      builder.onResolve({ filter: /^\.\/core\/runnerLifecycle$/ }, args => args.importer === join(source, "runnerMigration.ts") ? { path: "lifecycle", namespace: "duplicate-fixture" } : undefined);
      builder.onResolve({ filter: /^\.\/guided\/runnerStore$/ }, args => args.importer === join(source, "runnerMigration.ts") ? { path: "store", namespace: "duplicate-fixture" } : undefined);
      builder.onLoad({ filter: /.*/, namespace: "duplicate-fixture" }, args => {
        const bridge = "globalThis.__afDeploymentDuplicate", modules: Record<string, string> = {
          vscode: `const namespace = name => new Proxy({}, { get: (_, key) => ${bridge}.vscode[name][key] }); export const commands = namespace("commands"), window = namespace("window"), workspace = namespace("workspace"), Uri = namespace("Uri"), ViewColumn = namespace("ViewColumn"), ProgressLocation = namespace("ProgressLocation");`,
          azure: `export class AzureSession { constructor() { return ${bridge}.azure; } }`,
          lifecycle: `export * from ${JSON.stringify(lifecycle)}; import { submitRunner as actual } from ${JSON.stringify(lifecycle)}; export function submitRunner(...args) { ${bridge}.submitEntered(); return actual(...args); }`,
          store: `export * from ${JSON.stringify(store)}; import { RunnerStore as Actual } from ${JSON.stringify(store)}; export class RunnerStore extends Actual { async write(record) { await ${bridge}.beforeWrite(record); await super.write(record); await ${bridge}.afterWrite(record); } }`
        };
        assert.ok(modules[args.path]); return { contents: modules[args.path], loader: "js", resolveDir: source };
      });
    } }]
  });
  const inputs = Object.keys(result.metafile!.inputs); assert.ok(!inputs.some(path => /(?:^|\/)guided\/azure\.ts$/.test(path))); assert.equal(result.outputFiles.length, 1);
  const bytes = result.outputFiles[0]!.contents, bundleSHA256 = createHash("sha256").update(bytes).digest("hex");
  await retain(join(root, "runtime.cjs"), bytes);
  await retain(join(root, "package.json"), JSON.stringify({ name: "agefreighter-isolated-deployment-duplicate", publisher: "agefreighter-test-only", version: "0.0.0",
    displayName: "ISOLATED deployment duplicate fixture — no cloud", engines: { vscode: "^1.105.0" }, main: "./extension.cjs", activationEvents: ["onStartupFinished"], extensionKind: ["ui"],
    contributes: { commands: [{ command: "agefreighterFixture.openDeploymentDuplicate", title: "ISOLATED: Open deployment duplicate preview" }] } }, null, 2));
  await retain(join(root, "extension.cjs"), `"use strict";
const fs = require("node:fs/promises"), crypto = require("node:crypto");
exports.activate = async context => {
  try {
    if (crypto.createHash("sha256").update(await fs.readFile(__dirname + "/runtime.cjs")).digest("hex") !== ${JSON.stringify(bundleSHA256)}) throw Error("Duplicate fixture changed after preparation");
    await require("./runtime.cjs").activate(context);
  } catch(error) {
    const file = await fs.open(__dirname + "/activation-failed-" + process.pid + ".json", "wx", 0o600);
    try { await file.writeFile(JSON.stringify({pass: false, error: String(error), at: new Date().toISOString()})); await file.sync(); } finally { await file.close(); }
  }
};
`);
  await mkdir(join(root, "user-data"), { mode: 0o700 }); await mkdir(join(root, "user-data/User"), { mode: 0o700 }); await mkdir(join(root, "extensions"), { mode: 0o700 });
  await retain(join(root, "user-data/User/settings.json"), JSON.stringify({ "extensions.autoUpdate": false, "extensions.autoCheckUpdates": false }));
  const args = [`--extensionDevelopmentPath=${root}`, "--user-data-dir", join(root, "user-data"), "--extensions-dir", join(root, "extensions"), "--disable-extensions", "--skip-welcome", "--skip-release-notes", "--new-window"];
  const sourceHashes: Record<string, string> = {};
  for (const name of ["runnerMigration.ts", "core/runnerLifecycle.ts", "guided/runnerStore.ts", "test/helpers/nativeDeploymentDuplicate.ts", "test/helpers/nativeDeploymentDuplicateRuntime.ts", "test/helpers/nativeCancelOtherScenarios.ts"]) sourceHashes[name] = createHash("sha256").update(await readFile(join(source, name))).digest("hex");
  await retain(join(root, "preparation.json"), JSON.stringify({ mode, args, sourceHashes, inputs, bundleSHA256,
    secondWindowMustReuseExactProfileAndArgs: mode === "two-window", cloudQualification: false, requiresIndependentNativeInteractionObservation: true }, null, 2));
  return args;
}
