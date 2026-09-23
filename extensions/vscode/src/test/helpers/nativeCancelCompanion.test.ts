import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { prepareNativeCancelCompanion } from "./nativeCancelCompanion";

test("disposable companion uses normal development paths, no test mode, commands or install", async () => {
  const root = await mkdtemp("/private/tmp/af-native-cancel-");
  try {
    const extensionRoot = resolve(__dirname, "../../.."), args = await prepareNativeCancelCompanion(root, extensionRoot);
    assert.deepEqual(args.filter(arg => arg.startsWith("--extensionDevelopmentPath=")), [
      `--extensionDevelopmentPath=${extensionRoot}`, `--extensionDevelopmentPath=${root}/native-cancel-companion`]);
    assert.ok(!args.some(arg => /extensionTests|enable-proposed-api|install-extension|disable-workspace-trust/.test(arg)));
    const manifest = JSON.parse(await readFile(join(root, "native-cancel-companion/package.json"), "utf8"));
    assert.deepEqual(manifest.activationEvents, ["onStartupFinished"]); assert.equal(manifest.contributes, undefined);
    const main = await readFile(join(root, "native-cancel-companion/extension.cjs"), "utf8");
    assert.match(main, /require\(harness\)\.run\(context.extensionMode\)/); assert.doesNotMatch(main, /registerCommand|showWarningMessage|\.secrets|fetch\(/);
    assert.match(main, /Compiled native fixture changed after preparation/);
    assert.deepEqual(JSON.parse(await readFile(join(root, "user-data/User/settings.json"), "utf8")), {
      "extensions.autoUpdate": false, "extensions.autoCheckUpdates": false, "agefreighter.allowDevelopmentRunnerArtifacts": true });
    await assert.rejects(prepareNativeCancelCompanion(root, extensionRoot), /EEXIST/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("companion refuses a normal profile or arbitrary location before reading it", async () => {
  await assert.rejects(prepareNativeCancelCompanion("/not-a-disposable-profile", "unused"));
});
