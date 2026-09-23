/** Disposable development-only companion. Normal hosts show real native
 * dialogs; VS Code's extension-test mode intentionally suppresses them. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import { selectNativeCancelCases } from "./nativeCancelCatalog";

async function createFile(path: string, text: string): Promise<void> {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(text); await file.sync(); } finally { await file.close(); }
}

export async function prepareNativeCancelCompanion(root: string, extensionRoot: string, selection = ""): Promise<string[]> {
  const cases = selectNativeCancelCases(selection).map(s => s.id);
  assert.match(root, /^\/private\/tmp\/af-native-cancel-[a-zA-Z0-9]{6}$/);
  assert.equal(await realpath(root), root);
  const info = await lstat(root); assert.ok(info.isDirectory() && !info.isSymbolicLink()); assert.equal(info.mode & 0o777, 0o700);
  const harness = join(extensionRoot, "out-test/test/suite/nativeCancelHarness.js");
  const sha256 = createHash("sha256").update(await readFile(harness)).digest("hex");
  const companion = join(root, "native-cancel-companion");
  await mkdir(companion, { mode: 0o700 });
  await createFile(join(companion, "package.json"), JSON.stringify({
    name: "agefreighter-isolated-native-cancel", publisher: "agefreighter-test-only", version: "0.0.0",
    displayName: "ISOLATED native Cancel fixture — not a product extension", engines: { vscode: "^1.90.0" },
    main: "./extension.cjs", activationEvents: ["onStartupFinished"], extensionKind: ["ui"]
  }, null, 2));
  // No command registration, network client, secret access or automatic click.
  // Hash pin prevents a compiled harness changing after launch preparation.
  await createFile(join(companion, "extension.cjs"), `"use strict";
const fs = require("node:fs/promises"), crypto = require("node:crypto");
exports.activate = async function (context) {
  const root = ${JSON.stringify(root)}, harness = ${JSON.stringify(harness)};
  const retain = async (name, value) => { const f = await fs.open(root + "/" + name, "wx", 0o600); try { await f.writeFile(JSON.stringify(value, null, 2)); await f.sync(); } finally { await f.close(); } };
  try {
    if (crypto.createHash("sha256").update(await fs.readFile(harness)).digest("hex") !== ${JSON.stringify(sha256)}) throw Error("Compiled native fixture changed after preparation");
    process.env.AF_NATIVE_CANCEL_ROOT = root;
    process.env.AF_NATIVE_CANCEL_CASES = ${JSON.stringify(cases.join(","))};
    await require(harness).run(context.extensionMode);
    await retain("companion-completed.json", { finishedAt: new Date().toISOString(), limitation: "Harness returned; consult ledger.json for completed cases and pass status. Not signed-in cloud evidence." });
  } catch (error) {
    await retain("companion-failed.json", { pass: false, error: error instanceof Error ? error.message : "Unknown fixture failure", at: new Date().toISOString() });
    console.error("Isolated native cancellation fixture failed; see retained evidence.");
  }
};
`);
  await mkdir(join(root, "user-data"), { mode: 0o700 });
  await mkdir(join(root, "user-data/User"), { mode: 0o700 });
  await createFile(join(root, "user-data/User/settings.json"), JSON.stringify({ "extensions.autoUpdate": false, "extensions.autoCheckUpdates": false,
    "agefreighter.allowDevelopmentRunnerArtifacts": true }));
  return [`--extensionDevelopmentPath=${extensionRoot}`, `--extensionDevelopmentPath=${companion}`,
    "--user-data-dir", join(root, "user-data"), "--extensions-dir", join(root, "extensions"),
    "--disable-extensions", "--skip-welcome", "--skip-release-notes", "--new-window"];
}
