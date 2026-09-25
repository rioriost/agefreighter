import assert from "node:assert/strict";
import test from "node:test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, readdir, rm, symlink } from "node:fs/promises";
import { join, resolve } from "node:path";

const helper = resolve(__dirname, "../../../scripts/native-lock-fixture.cjs");
const invoke = (...args: string[]) => promisify(execFile)(process.execPath, [helper, ...args], { timeout: 20_000, maxBuffer: 32_768 });

test("native fixture helper prepares only an exited own-child lock and cancellation verification is create-only", { skip: process.platform !== "darwin" }, async () => {
  const prepared = JSON.parse((await invoke("prepare")).stdout);
  assert.match(prepared.root, /^\/private\/tmp\/af-native-lock-[a-zA-Z0-9]{6}$/);
  const root: string = prepared.root;
  try {
    const baseline = JSON.parse(await readFile(join(root, "baseline.json"), "utf8"));
    const marker = JSON.parse(await readFile(join(root, "child-exit-marker.json"), "utf8"));
    assert.equal(marker.pid, prepared.childPid); assert.equal(marker.exitCode, 23);
    assert.deepEqual(baseline.childExit, { code: 23, signal: null });
    assert.ok(prepared.launchArguments.includes(`--extensionDevelopmentPath=${resolve(__dirname, "../../..")}`));
    assert.ok(prepared.launchArguments.includes(join(root, "user-data")));
    const userSettings = JSON.parse(await readFile(join(root, "user-data/User/settings.json"), "utf8"));
    assert.equal(userSettings["extensions.autoUpdate"], false);
    assert.equal(userSettings["extensions.autoCheckUpdates"], false);
    const workspace = JSON.parse(await readFile(join(root, "B10-ISOLATED.code-workspace"), "utf8"));
    assert.equal(Object.hasOwn(workspace.settings, "extensions.autoUpdate"), false);
    assert.equal(Object.hasOwn(workspace.settings, "extensions.autoCheckUpdates"), false);
    assert.equal(Object.hasOwn(workspace.settings, "security.workspace.trust.enabled"), false, "Keep the native trust default without workspace overrides");
    const store = join(root, "user-data/User/globalStorage/rioriost.agefreighter/runner-v2");
    const before = (await readdir(store)).sort();
    const cancelled = JSON.parse((await invoke("verify-cancel", root)).stdout);
    assert.equal(cancelled.pass, true); assert.equal(cancelled.mode, "verify-cancel");
    assert.deepEqual((await readdir(store)).sort(), before);
    const evidence = await readFile(join(root, "verify-cancel.json"));
    await assert.rejects(invoke("verify-cancel", root), /EEXIST/);
    assert.deepEqual(await readFile(join(root, "verify-cancel.json")), evidence);
    await assert.rejects(invoke("verify-recovered", root), /Exactly one native recovery archive/);
    assert.deepEqual((await readdir(store)).sort(), before, "A failed check cannot recover the lock");
    // A linked file in the fixture is refused before report/store access.
    const baselinePath = join(root, "baseline.json"), snapshot = join(root, "snapshots", baseline.recordFile);
    await rm(baselinePath); await symlink(snapshot, baselinePath);
    await assert.rejects(invoke("verify-cancel", root), /regular unlinked fixture file/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("native fixture verifier rejects arbitrary paths and invalid modes before accessing them", { skip: process.platform !== "darwin" }, async () => {
  await assert.rejects(invoke("verify-cancel", "/Users/not-a-fixture"), /fresh dedicated/);
  await assert.rejects(invoke("verify-recovered", "/private/tmp/af-native-lock-ABCDEF/../elsewhere"), /fresh dedicated/);
  await assert.rejects(invoke("recover", "/private/tmp/af-native-lock-ABCDEF"), /Usage/);
});
