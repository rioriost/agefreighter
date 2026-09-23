/* Test-only fixture utility; scripts/** is excluded from the VSIX.
 * No Azure adapters, installed-profile discovery, UI automation or recovery
 * approval. Only a real native command interaction can recover this fixture. */
"use strict";

require("tsx/cjs");
const assert = require("node:assert/strict");
const { createHash, randomUUID } = require("node:crypto");
const { spawn } = require("node:child_process");
const { lstat, mkdir, mkdtemp, open, readFile, readdir, realpath, stat } = require("node:fs/promises");
const { dirname, join, resolve, relative } = require("node:path");
const { RunnerStore } = require("../src/guided/runnerStore.ts");
const { sourceWorkflowDraft } = require("../src/core/runner.ts");

const extensionRoot = resolve(__dirname, "..");
const prefix = "/private/tmp/af-native-lock-";
const rootPattern = /^\/private\/tmp\/af-native-lock-[a-zA-Z0-9]{6}$/;
const uuidPattern = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const sha = data => createHash("sha256").update(data).digest("hex");
const storePath = root => join(root, "user-data/User/globalStorage/rioriost.agefreighter/runner-v2");

async function regular(path) {
  const info = await lstat(path);
  assert.ok(info.isFile() && !info.isSymbolicLink(), "Expected a regular unlinked fixture file");
  return info;
}
async function checkedRoot(root) {
  assert.equal(process.platform, "darwin", "This native fixture is scoped to the current macOS desktop");
  assert.ok(typeof root === "string" && rootPattern.test(root), "Only a fresh dedicated /private/tmp/af-native-lock-* fixture is accepted");
  assert.equal(await realpath(root), root, "Fixture root cannot use a linked path");
  const info = await lstat(root);
  assert.ok(info.isDirectory() && !info.isSymbolicLink());
  assert.equal(info.uid, process.getuid()); assert.equal(info.mode & 0o777, 0o700);
}
async function checkedPath(root, path) {
  const local = relative(root, path);
  assert.ok(local && !local.startsWith("..") && !local.startsWith("/"), "Fixture path escapes its dedicated root");
  let current = dirname(path);
  while (current !== root) {
    const info = await lstat(current);
    assert.ok(info.isDirectory() && !info.isSymbolicLink(), "Linked fixture ancestor refused");
    current = dirname(current);
  }
  return regular(path);
}
async function createOnly(root, relativePath, bytes) {
  const path = join(root, relativePath), parent = dirname(path);
  // Every parent is inside the newly created fixture; no external path is used.
  assert.ok(path.startsWith(root + "/") && !relativePath.split("/").includes(".."));
  await mkdir(parent, { recursive: true, mode: 0o700 });
  let current = parent;
  while (current !== root) {
    const info = await lstat(current);
    assert.ok(info.isDirectory() && !info.isSymbolicLink(), "Linked fixture write ancestor refused");
    current = dirname(current);
  }
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
  const directory = await open(parent, "r"); try { await directory.sync(); } finally { await directory.close(); }
}
const saveJSON = (root, name, value) => createOnly(root, name, JSON.stringify(value, null, 2) + "\n");
async function readJSON(root, name) {
  const path = join(root, name); await checkedPath(root, path);
  return JSON.parse(await readFile(path, "utf8"));
}
async function pin() {
  const files = ["dist/extension.js", "package.json"];
  return Object.fromEntries(await Promise.all(files.map(async name => {
    const path = join(extensionRoot, name); await regular(path); return [name, sha(await readFile(path))];
  })));
}

async function ownChild(root, workflow, nonce) {
  await checkedRoot(root);
  const intent = await readJSON(root, "fixture-intent.json");
  assert.equal(intent.kind, "isolated-native-runner-lock"); assert.equal(intent.root, root);
  assert.equal(intent.parentPid, process.ppid); assert.equal(intent.nonce, nonce); assert.equal(intent.workflow, workflow);
  assert.ok(uuidPattern.test(workflow)); assert.deepEqual(await pin(), intent.extensionPin);
  const store = new RunnerStore(storePath(root));
  await store.exclusive(workflow, async () => {
    const lockPath = join(storePath(root), `${workflow}.lock`); await checkedPath(root, lockPath);
    const owner = JSON.parse(await readFile(lockPath, "utf8")); assert.equal(owner.pid, process.pid);
    await saveJSON(root, "child-exit-marker.json", { kind: intent.kind, root, workflow, nonce, pid: process.pid,
      parentPid: process.ppid, exitCode: 23, lockSHA256: sha(await readFile(lockPath)), createdAt: new Date().toISOString() });
    // Abruptly exit only this known child, skipping RunnerStore's finally.
    process.exit(23);
  });
  throw new Error("Fixture child did not exit inside its own protected action");
}

async function prepare() {
  assert.equal(process.platform, "darwin", "macOS-only native fixture");
  const extensionPin = await pin();
  const root = await mkdtemp(prefix); await checkedRoot(root);
  const workflow = randomUUID(), operation = randomUUID(), nonce = randomUUID();
  await saveJSON(root, "fixture-intent.json", { version: 1, kind: "isolated-native-runner-lock", root, workflow, nonce,
    parentPid: process.pid, extensionRoot, extensionPin, createdAt: new Date().toISOString() });
  const store = new RunnerStore(storePath(root));
  const record = sourceWorkflowDraft(workflow, { subscriptionId: workflow, resourceGroup: "b10-isolated-local-fixture",
    region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused-local-fixture",
    source: { type: "csv", location: "local" } });
  const reportText = JSON.stringify({ kind: "isolated-native-lock-report", synthetic: true, workflow, operation,
    message: "Harmless local fixture. No source data, account or cloud operation." });
  const report = { operation, bytes: Buffer.byteLength(reportText), sha256: sha(reportText) };
  await store.write(record); await store.retainReport(workflow, report, reportText); await store.syncEvidenceDirectory();
  const child = spawn(process.execPath, [__filename, "--own-child", root, workflow, nonce], {
    cwd: extensionRoot, stdio: ["ignore", "pipe", "pipe"], timeout: 10_000, killSignal: "SIGKILL"
  });
  let output = "";
  child.stdout.on("data", data => { output = (output + data.toString()).slice(-4096); });
  child.stderr.on("data", data => { output = (output + data.toString()).slice(-4096); });
  const result = await new Promise((resolveResult, reject) => {
    child.once("error", reject); child.once("exit", (code, signal) => resolveResult({ code, signal }));
  });
  assert.equal(result.code, 23, `Known fixture child did not exit as expected: ${output}`); assert.equal(result.signal, null);
  const marker = await readJSON(root, "child-exit-marker.json");
  assert.equal(marker.pid, child.pid); assert.equal(marker.parentPid, process.pid); assert.equal(marker.nonce, nonce);
  const review = await store.reviewCrashLock(workflow); assert.equal(review.owner.pid, child.pid);
  await store.recoverCrashLock(review, false); // Cancel only; never approve through this utility.
  const recordFile = `${workflow}.json`, reportFile = `${workflow}.report-${operation}.json`, lockFile = `${workflow}.lock`;
  const files = {};
  for (const name of [recordFile, reportFile, lockFile]) {
    const path = join(storePath(root), name), info = await checkedPath(root, path), bytes = await readFile(path);
    files[name] = { sha256: sha(bytes), bytes: bytes.length, device: info.dev, inode: info.ino };
    await createOnly(root, `snapshots/${name}`, bytes);
  }
  assert.equal(files[lockFile].sha256, marker.lockSHA256);
  await mkdir(join(root, "extensions"), { mode: 0o700 });
  await mkdir(join(root, "workspace"), { mode: 0o700 });
  const banner = `# ISOLATED B10 LOCAL LOCK FIXTURE\n\nThis window uses a disposable VS Code profile.\n\n- Fixture root: ${root}\n- Workflow: ${workflow}\n- Former fixture process: ${child.pid} (confirmed exit 23)\n- No Azure account, source credentials, migration or remote command is configured.\n- This is a native local-confirmation test, not signed-in active-cloud crash qualification.\n\nRun **AGEFreighter: Review Interrupted Runner Lock** from the Command Palette.\nSelect the only workflow, inspect the native confirmation, and first choose Cancel.\nAfter external cancellation verification, repeat and approve Recover local lock.\nOnly the native dialog may approve recovery; the fixture utility does not.\n`;
  await createOnly(root, "workspace/ISOLATED-B10-READ-ME.md", banner);
  // VS Code permits extension update controls only in user settings, not in a
  // workspace. This file belongs exclusively to the fresh disposable profile.
  await saveJSON(root, "user-data/User/settings.json", {
    "extensions.autoUpdate": false, "extensions.autoCheckUpdates": false
  });
  await saveJSON(root, "B10-ISOLATED.code-workspace", { folders: [{ name: "B10 ISOLATED LOCAL FIXTURE", path: "workspace" }], settings: {
    "window.title": "B10 ISOLATED LOCAL FIXTURE — ${activeEditorShort}",
    "workbench.startupEditor": "none",
    "workbench.colorCustomizations": { "titleBar.activeBackground": "#5e3215", "titleBar.activeForeground": "#ffffff" }
  } });
  const launchArguments = ["--new-window", "--user-data-dir", join(root, "user-data"), "--extensions-dir", join(root, "extensions"),
    "--disable-extensions", "--skip-welcome", "--skip-release-notes", `--extensionDevelopmentPath=${extensionRoot}`,
    join(root, "B10-ISOLATED.code-workspace"), join(root, "workspace/ISOLATED-B10-READ-ME.md")];
  const baseline = { version: 1, kind: "isolated-native-runner-lock", root, workflow, report, recordFile, reportFile, lockFile,
    childPid: child.pid, childExit: result, extensionRoot, extensionPin, files, launchArguments, createdAt: new Date().toISOString() };
  await saveJSON(root, "baseline.json", baseline);
  console.log(JSON.stringify({ root, workflow, childPid: child.pid, extensionPin, launchArguments,
    verification: { cancel: [process.execPath, __filename, "verify-cancel", root], recovered: [process.execPath, __filename, "verify-recovered", root] } }, null, 2));
}

async function verify(mode, root) {
  await checkedRoot(root);
  const baseline = await readJSON(root, "baseline.json"), intent = await readJSON(root, "fixture-intent.json");
  assert.equal(baseline.version, 1); assert.equal(baseline.kind, "isolated-native-runner-lock"); assert.equal(baseline.root, root);
  assert.equal(intent.root, root); assert.equal(intent.workflow, baseline.workflow); assert.equal(intent.kind, baseline.kind);
  assert.ok(uuidPattern.test(baseline.workflow) && uuidPattern.test(baseline.report.operation));
  assert.equal(baseline.extensionRoot, extensionRoot); assert.deepEqual(await pin(), baseline.extensionPin, "Pinned compiled extension changed");
  assert.equal(baseline.recordFile, `${baseline.workflow}.json`);
  assert.equal(baseline.reportFile, `${baseline.workflow}.report-${baseline.report.operation}.json`);
  assert.equal(baseline.lockFile, `${baseline.workflow}.lock`);
  const storeRoot = storePath(root), storeInfo = await lstat(storeRoot);
  assert.ok(storeInfo.isDirectory() && !storeInfo.isSymbolicLink());
  const assertOriginal = async name => {
    const path = join(storeRoot, name), info = await checkedPath(root, path), bytes = await readFile(path);
    const snapshot = join(root, "snapshots", name); await checkedPath(root, snapshot);
    const original = await readFile(snapshot);
    assert.equal(sha(original), baseline.files[name].sha256, "Original snapshot changed");
    assert.equal(sha(bytes), baseline.files[name].sha256, "Fixture content changed"); assert.deepEqual(bytes, original);
    return info;
  };
  await assertOriginal(baseline.recordFile); await assertOriginal(baseline.reportFile);
  const store = new RunnerStore(storeRoot);
  assert.equal(sha(await store.readReport(baseline.workflow, baseline.report)), baseline.report.sha256);
  const entries = (await readdir(storeRoot)).sort();
  let archiveEvidence;
  if (mode === "verify-cancel") {
    assert.deepEqual(entries, [baseline.recordFile, baseline.reportFile, baseline.lockFile].sort(), "Cancellation must create no store entries");
    const info = await assertOriginal(baseline.lockFile);
    assert.equal(info.dev, baseline.files[baseline.lockFile].device); assert.equal(info.ino, baseline.files[baseline.lockFile].inode);
    const review = await store.reviewCrashLock(baseline.workflow); assert.equal(review.owner.pid, baseline.childPid);
    await store.recoverCrashLock(review, false);
  } else {
    const archives = entries.filter(name => new RegExp(`^${baseline.workflow}\\.recovered-lock-[a-f0-9-]{36}\\.json$`).test(name));
    assert.equal(archives.length, 1, "Exactly one native recovery archive is required");
    assert.deepEqual(entries, [baseline.recordFile, baseline.reportFile, archives[0]].sort());
    await assert.rejects(stat(join(storeRoot, baseline.lockFile)), { code: "ENOENT" });
    const archivePath = join(storeRoot, archives[0]); await checkedPath(root, archivePath);
    const bytes = await readFile(archivePath), archive = JSON.parse(bytes.toString("utf8"));
    assert.equal(archive.version, 1); assert.equal(archive.kind, "operator-reviewed-runner-lock");
    assert.equal(archive.scope, "local-lock-only-no-remote-action"); assert.equal(archive.review.workflowId, baseline.workflow);
    assert.equal(archive.review.owner.pid, baseline.childPid); assert.equal(archive.review.lockSHA256, baseline.files[baseline.lockFile].sha256);
    assert.equal(archive.review.recordSHA256, baseline.files[baseline.recordFile].sha256);
    const snapshot = join(root, "snapshots", baseline.lockFile); await checkedPath(root, snapshot);
    assert.equal(archive.originalLock, await readFile(snapshot, "utf8"));
    assert.equal(sha(archive.originalLock), baseline.files[baseline.lockFile].sha256);
    assert.equal(archive.review.device, baseline.files[baseline.lockFile].device); assert.equal(archive.review.inode, baseline.files[baseline.lockFile].inode);
    assert.deepEqual(JSON.parse(archive.originalLock), archive.review.owner);
    // A local read under the production gate proves usability; no recovery or
    // remote controller is invoked, and the action never writes the workflow.
    await store.exclusive(baseline.workflow, () => store.read(baseline.workflow));
    await assertOriginal(baseline.recordFile); await assertOriginal(baseline.reportFile);
    assert.deepEqual((await readdir(storeRoot)).sort(), entries);
    archiveEvidence = { name: archives[0], sha256: sha(bytes), bytes: bytes.length, approvedAt: archive.approvedAt };
  }
  const evidence = { version: 1, kind: "isolated-native-runner-lock-verification", mode, pass: true, root, workflow: baseline.workflow,
    childPid: baseline.childPid, extensionPin: baseline.extensionPin, recordSHA256: baseline.files[baseline.recordFile].sha256,
    reportSHA256: baseline.files[baseline.reportFile].sha256, originalLockSHA256: baseline.files[baseline.lockFile].sha256,
    ...(archiveEvidence ? { archive: archiveEvidence } : {}), checkedAt: new Date().toISOString(),
    limitation: "File evidence only. Root task must separately record actual native UI interaction. No signed-in active-cloud crash qualification." };
  await saveJSON(root, `${mode}.json`, evidence);
  console.log(JSON.stringify(evidence, null, 2));
}

async function main() {
  const [mode, root, workflow, nonce] = process.argv.slice(2);
  if (mode === "prepare" && process.argv.length === 3) return prepare();
  if (mode === "--own-child" && process.argv.length === 6) return ownChild(root, workflow, nonce);
  if ((mode === "verify-cancel" || mode === "verify-recovered") && process.argv.length === 4) return verify(mode, root);
  throw new Error("Usage: native-lock-fixture.cjs prepare | verify-cancel FIXTURE_ROOT | verify-recovered FIXTURE_ROOT");
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
