import * as path from "node:path";
import { runTests } from "@vscode/test-electron";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../..");
  const extensionTestsPath = path.resolve(__dirname, "suite", "index");
  // Never launch qualification tests against the operator's Azure-signed-in
  // profile or its SecretStorage. Retain the disposable profile for diagnostics.
  const profile = await mkdtemp(path.join(process.platform === "darwin" ? "/tmp" : tmpdir(), "af-extension-host-"));
  console.log(`Isolated Extension Host profile: ${profile}`);
  await runTests({
    version: process.env.VSCODE_TEST_VERSION ?? "1.105.0",
    ...(process.env.VSCODE_TEST_EXECUTABLE ? {vscodeExecutablePath: process.env.VSCODE_TEST_EXECUTABLE} : {}),
    extensionDevelopmentPath,
    extensionTestsPath,
    extensionTestsEnv: { AF_ISOLATED_HOST_ROOT: profile },
    launchArgs: ["--user-data-dir", path.join(profile, "user-data"),
      "--extensions-dir", path.join(profile, "extensions"), "--disable-extensions",
      "--skip-welcome", "--skip-release-notes", "--disable-workspace-trust"]
  });
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
