import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { runTests } from "@vscode/test-electron";

async function main(): Promise<void> {
  // Never use the operator's signed-in profile or an installed extension store.
  // macOS's long per-user tmpdir can exceed the Unix-domain IPC path limit.
  const root = await mkdtemp(join(process.platform === "darwin" ? "/tmp" : tmpdir(), "af-host-crash-"));
  console.log(`Retained isolated crash evidence: ${root}`);
  for (const point of ["before-dispatch", "after-dispatch"]) {
    const evidence = join(root, point);
    const options = {
      version: process.env.VSCODE_TEST_VERSION ?? "1.138.0",
      ...(process.env.VSCODE_TEST_EXECUTABLE ? { vscodeExecutablePath: process.env.VSCODE_TEST_EXECUTABLE } : {}),
      extensionDevelopmentPath: resolve(__dirname, "../.."),
      extensionTestsPath: resolve(__dirname, "suite/crashHarness"),
      launchArgs: ["--user-data-dir", join(root, point, "profile"), "--extensions-dir", join(root, point, "extensions"),
        "--disable-extensions", "--skip-welcome", "--skip-release-notes", "--disable-workspace-trust"],
    };
    let rejected = false;
    try {
      await runTests({ ...options, extensionTestsEnv: { AF_CRASH_ROOT: evidence, AF_CRASH_PHASE: "crash", AF_CRASH_POINT: point } });
    } catch { rejected = true; }
    assert.equal(rejected, true, "SIGKILL must not appear as a successful test run");
    const marker = JSON.parse(await readFile(join(evidence, "crash.json"), "utf8"));
    assert.equal(marker.point, point);
    assert.equal(marker.signal, "SIGKILL");
    assert.ok(Number.isSafeInteger(marker.pid) && marker.pid > 0);
    // Independent parent-side VS Code evidence, not merely the pre-kill marker.
    const logRoot = join(evidence, "profile/logs");
    const logs = await Promise.all((await readdir(logRoot)).map(async dir => {
      try { return await readFile(join(logRoot, dir, "main.log"), "utf8"); }
      catch { return ""; }
    }));
    assert.ok(logs.some(log => new RegExp(`Extension host with pid ${marker.pid} exited with code: (?:9, signal: unknown|[^\\n]*signal: SIGKILL)`).test(log)),
      "VS Code must independently record this exact host's forced exit");
    await runTests({ ...options, extensionTestsEnv: { AF_CRASH_ROOT: evidence, AF_CRASH_PHASE: "recover", AF_CRASH_POINT: point } });
    const result = JSON.parse(await readFile(join(evidence, "recovery.json"), "utf8"));
    assert.equal(result.pass, true);
    assert.notEqual(result.pid, marker.pid, "Recovery must run in a different Extension Host");
    console.log(JSON.stringify(result));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
