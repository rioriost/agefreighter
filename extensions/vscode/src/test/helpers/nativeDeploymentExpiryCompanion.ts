/** Build only: never opens VS Code, touches operator state or calls Azure. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import { build } from "esbuild";

async function createFile(path: string, bytes: string | Uint8Array) {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
}
const hash = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");

export async function prepareDeploymentExpiryCompanion(root: string, extensionRoot: string): Promise<string[]> {
  assert.match(root, /^\/private\/tmp\/af-deployment-expiry-[a-zA-Z0-9]{6}$/);
  assert.equal(await realpath(root), root);
  const info = await lstat(root); assert.ok(info.isDirectory() && !info.isSymbolicLink()); assert.equal(info.mode & 0o777, 0o700);
  // Acquire the single-use preparation marker before reading/building anything.
  await createFile(join(root, "preparing.json"), JSON.stringify({ at: new Date().toISOString(), expiryStartsAtActivation: true }));
  const source = join(extensionRoot, "src"), lifecycle = join(source, "core/runnerLifecycle.ts"), store = join(source, "guided/runnerStore.ts");
  const result = await build({
    entryPoints: [join(source, "test/helpers/nativeDeploymentExpiryRuntime.ts")], bundle: true, platform: "node", format: "cjs", target: "node22",
    write: false, metafile: true, logLevel: "silent", external: ["vscode"],
    define: { fetch: "globalThis.__afDeploymentExpiry.denyFetch" },
    plugins: [{ name: "expiry-inert-boundaries", setup(builder) {
      builder.onResolve({ filter: /^vscode$/ }, args => args.kind === "import-statement" ? { path: "vscode", namespace: "expiry-fixture" } : undefined);
      builder.onResolve({ filter: /\/guided\/azure$/ }, () => ({ path: "azure", namespace: "expiry-fixture" }));
      builder.onResolve({ filter: /^\.\/core\/runnerLifecycle$/ }, args => args.importer === join(source, "runnerMigration.ts") ? { path: "lifecycle", namespace: "expiry-fixture" } : undefined);
      builder.onResolve({ filter: /^\.\/guided\/runnerStore$/ }, args => args.importer === join(source, "runnerMigration.ts") ? { path: "store", namespace: "expiry-fixture" } : undefined);
      builder.onLoad({ filter: /.*/, namespace: "expiry-fixture" }, args => {
        const bridge = "globalThis.__afDeploymentExpiry";
        const modules: Record<string, string> = {
          // esbuild's CJS-to-ESM wrapper enumerates exports, so a bare empty
          // namespace proxy would lose all members. Explicit lazy namespaces
          // keep initialization inert and preserve production property access.
          vscode: `const namespace = name => new Proxy({}, { get: (_, key) => ${bridge}.vscode[name][key] }); export const commands = namespace("commands"), window = namespace("window"), workspace = namespace("workspace"), Uri = namespace("Uri"), ViewColumn = namespace("ViewColumn"), ProgressLocation = namespace("ProgressLocation");`,
          azure: `export class AzureSession { constructor() { return ${bridge}.azure; } }`,
          lifecycle: `export * from ${JSON.stringify(lifecycle)}; import { submitRunner as actual } from ${JSON.stringify(lifecycle)}; export function submitRunner(...args) { ${bridge}.submitEntered(); return actual(...args); }`,
          store: `export * from ${JSON.stringify(store)}; import { RunnerStore as Actual } from ${JSON.stringify(store)}; export class RunnerStore extends Actual { async write(record) { ${bridge}.storeWrite(); return super.write(record); } }`
        };
        assert.ok(modules[args.path]); return { contents: modules[args.path], loader: "js", resolveDir: source };
      });
    } }]
  });
  assert.equal(result.outputFiles.length, 1);
  // A real account adapter must not enter this bundle, even transitively.
  const inputs = Object.keys(result.metafile!.inputs);
  assert.ok(!inputs.some(path => /(?:^|\/)guided\/azure\.ts$/.test(path)), "Real Azure adapter was bundled");
  const bytes = result.outputFiles[0]!.contents, bundleSHA256 = hash(bytes);
  await createFile(join(root, "runtime.cjs"), bytes);
  await createFile(join(root, "package.json"), JSON.stringify({
    name: "agefreighter-isolated-deployment-expiry", publisher: "agefreighter-test-only", version: "0.0.0",
    displayName: "ISOLATED deployment expiry fixture — no cloud", engines: { vscode: "^1.105.0" },
    main: "./extension.cjs", activationEvents: ["onStartupFinished"], extensionKind: ["ui"],
    contributes: { commands: [{ command: "agefreighterFixture.openDeploymentExpiry", title: "ISOLATED: Open deployment expiry preview" }] }
  }, null, 2));
  await createFile(join(root, "extension.cjs"), `"use strict";
const fs = require("node:fs/promises"), crypto = require("node:crypto");
exports.activate = async context => {
  try {
    const bytes = await fs.readFile(__dirname + "/runtime.cjs");
    if (crypto.createHash("sha256").update(bytes).digest("hex") !== ${JSON.stringify(bundleSHA256)}) throw Error("Expiry fixture bundle changed after preparation");
    await require("./runtime.cjs").activate(context);
  } catch (error) {
    const file = await fs.open(__dirname + "/activation-failed.json", "wx", 0o600);
    try { await file.writeFile(JSON.stringify({ pass: false, error: String(error), at: new Date().toISOString() })); await file.sync(); } finally { await file.close(); }
  }
};
`);
  await mkdir(join(root, "user-data"), { mode: 0o700 });
  await mkdir(join(root, "user-data/User"), { mode: 0o700 });
  await mkdir(join(root, "extensions"), { mode: 0o700 });
  await createFile(join(root, "user-data/User/settings.json"), JSON.stringify({ "extensions.autoUpdate": false, "extensions.autoCheckUpdates": false }));
  const args = [`--extensionDevelopmentPath=${root}`, "--user-data-dir", join(root, "user-data"), "--extensions-dir", join(root, "extensions"),
    "--disable-extensions", "--skip-welcome", "--skip-release-notes", "--new-window"];
  const sourceHashes: Record<string, string> = {};
  for (const name of ["runnerMigration.ts", "core/runnerLifecycle.ts", "guided/runnerStore.ts", "test/helpers/nativeDeploymentExpiry.ts", "test/helpers/nativeDeploymentExpiryRuntime.ts", "test/helpers/nativeCancelOtherScenarios.ts"]) {
    sourceHashes[name] = hash(await readFile(join(source, name)));
  }
  await createFile(join(root, "preparation.json"), JSON.stringify({ preparedAt: new Date().toISOString(), bundleSHA256, sourceHashes, args,
    inputs, expiryStartsAtActivation: true, evidenceLayer: "isolated-normal-native-production-controller-with-inert-effects", signedInCloudQualification: false }, null, 2));
  return args;
}
