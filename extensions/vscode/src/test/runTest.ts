import * as path from "node:path";
import { runTests } from "@vscode/test-electron";

async function main(): Promise<void> {
  const extensionDevelopmentPath = path.resolve(__dirname, "../..");
  const extensionTestsPath = path.resolve(__dirname, "suite", "index");
  await runTests({
    version: process.env.VSCODE_TEST_VERSION ?? "1.105.0",
    extensionDevelopmentPath,
    extensionTestsPath
  });
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
