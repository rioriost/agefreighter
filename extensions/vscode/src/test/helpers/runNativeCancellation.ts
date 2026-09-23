/** Root-operated manual launch only; never included in automatic test scripts.
 * First compile with npm run compile && npm run test:host:compile.
 * Then: VSCODE_TEST_EXECUTABLE='/exact/isolated/Code' npx tsx
 * src/test/helpers/runNativeCancellation.ts
 * Actual modal clicks must be made by the operator/root; no mock cancellation. */
import assert from "node:assert/strict";
import { mkdtemp, open } from "node:fs/promises";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { prepareNativeCancelCompanion } from "./nativeCancelCompanion";
import { selectNativeCancelCases } from "./nativeCancelCatalog";

async function main(): Promise<void> {
  assert.equal(process.platform, "darwin", "This initial native fixture runner is macOS-scoped");
  const executable = process.env.VSCODE_TEST_EXECUTABLE;
  assert.ok(typeof executable === "string" && executable.startsWith("/"), "Set the exact reviewed isolated VS Code executable; do not download or install one");
  const option = process.argv.slice(2);
  assert.ok(option.length === 0 || option.length === 1 && option[0]!.startsWith("--cases="), "Optional argument: --cases=A01,A02,... (unique frozen IDs)");
  const selection = option[0]?.slice("--cases=".length) ?? "";
  const cases = selectNativeCancelCases(selection).map(s => s.id);
  const root = await mkdtemp("/private/tmp/af-native-cancel-");
  const extensionDevelopmentPath = resolve(__dirname, "../../..");
  const args = await prepareNativeCancelCompanion(root, extensionDevelopmentPath, selection);
  const metadata = await open(join(root, "launch.json"), "wx", 0o600);
  try { await metadata.writeFile(JSON.stringify({ root, executable, extensionDevelopmentPath, args, cases, launchMode: "normal-development-host-with-disposable-companion", syntheticPrerequisites: true, actualNativeCancelRequired: true })); await metadata.sync(); }
  finally { await metadata.close(); }
  console.log(`Retained native cancellation fixture root: ${root}`);
  const child = spawn(executable, args, { stdio: "inherit" });
  await new Promise<void>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => code === 0 ? resolve() : reject(new Error(`Development host exited: code=${code}, signal=${signal}`)));
  });
}
void main().catch(error => { console.error(error); process.exitCode = 1; });
