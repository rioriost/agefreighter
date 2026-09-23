import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { bootstrapScript, RunnerRecord } from "../../core/runner";
import { dispatchGuest, guestReadinessScript, reconcileGuest } from "../../core/runnerGuest";
import { ensureAssessmentReadiness, startAssessment } from "../../core/runnerAssessment";
import { RunnerControl } from "../../core/runnerLifecycle";

// Local shell/controller regression only. Neither Azure nor cloud-init is run.
test("missing tools member fails the bootstrap extraction before any executable install", {skip: process.platform === "win32"}, () => {
  const dir = mkdtempSync(join(tmpdir(), "af-b09-packaging-"));
  try {
    writeFileSync(join(dir, "agefreighter"), "inert fixture: must never execute\n");
    const archive = join(dir, "negative.tar.gz");
    assert.equal(spawnSync("tar", ["-czf", archive, "-C", dir, "agefreighter"], {env: {...process.env, COPYFILE_DISABLE: "1"}}).status, 0);
    const sha256 = createHash("sha256").update(readFileSync(archive)).digest("hex");
    const script = bootstrapScript({version: "2.4.0", sha256, url: "https://github.com/rioriost/agefreighter/releases/download/v2.4.0/agefreighter_v2.4.0_linux_amd64.tar.gz"});
    const installDirectory = join(dir, "state"), executables = join(dir, "bin");
    const local = script.replace(/^curl .+$/m, `cp '${archive}' "$work/archive.tar.gz"`)
      .replaceAll("/var/lib/agefreighter", installDirectory).replaceAll("/usr/local/bin", executables);
    assert.ok(!local.includes("curl ") && !local.includes("/var/lib/") && !local.includes("/usr/local/bin"));
    // macOS sha256sum lacks GNU long options; preserve a real checksum verification.
    const checksumAdapter = process.platform === "darwin" ? "sha256sum() { [ \"$*\" = '--check --status' ] || return 99; shasum -a 256 --check --status; }\n" : "";
    const result = spawnSync("bash", ["-c", checksumAdapter + local], {encoding: "utf8", timeout: 5000});
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /agefreighter-tools/);
    assert.equal(existsSync(join(executables, "agefreighter")), false);
    assert.equal(existsSync(join(executables, "agefreighter-tools")), false);
    assert.equal(existsSync(join(installDirectory, "bootstrap.complete")), false);
    assert.equal(existsSync(join(installDirectory, "evidence", "archive.sha256")), false);
    assert.equal(existsSync(join(installDirectory, "evidence", "version.txt")), false);
  } finally { rmSync(dir, {recursive: true, force: true}); }
});

test("terminal cloud-init failure exits readiness before installation checks or dispatch", {skip: process.platform === "win32"}, () => {
  // The production shell must exit before these absolute paths can be reached.
  const safe = guestReadinessScript.replaceAll("/var/lib/agefreighter/bootstrap.complete", "/nonexistent-af-b09-marker")
    .replaceAll("/usr/local/bin/agefreighter-tools", "/nonexistent-af-b09-tool");
  const result = spawnSync("bash", ["-c", `timeout() { [ "$*" = '45 cloud-init status --wait' ] || return 99; return 1; }\n${safe}`], {encoding: "utf8", timeout: 5000, env: {...process.env, AF_RUNNER_REQUEST: ""}});
  assert.equal(result.status, 1);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr.trim(), "Linux bootstrap did not complete successfully.");
});

test("failed real-command-shaped readiness result remains unready and all source gates emit no effects", async () => {
  const id = "11111111-1111-4111-8111-111111111111", operation = "22222222-2222-4222-8222-222222222222";
  const record: RunnerRecord = {
    schemaVersion: 2, id, phase: "provisioned",
    input: {subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", subnetId: "subnet", size: "Standard_B2s_v2", source: {type: "neo4j", location: "on-premises"}},
    artifact: {version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/artifact"},
    vmId: `/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/runner`,
    deploymentId: "deployment", template: {}, previewHash: "hash", expiresAt: "", updatedAt: "", hourlyComputeUSD: .1
  };
  const effects: string[] = [], saved: RunnerRecord[] = [];
  let submitted = false;
  const control: RunnerControl = {
    sleep: async () => { effects.push("sleep"); },
    list: async () => { effects.push("list"); return []; },
    persist: async r => { effects.push("persist"); saved.push(structuredClone(r)); },
    request: async (_subscription, path, method = "GET") => {
      effects.push(`${method}:${path}`);
      if (method === "PUT") { submitted = true; return {status: 201, value: {}}; }
      if (!submitted) return {status: 404, value: {}};
      return {status: 200, value: {properties: {provisioningState: "Succeeded", instanceView: {executionState: "Failed", exitCode: 1, error: "Linux bootstrap did not complete successfully."}}}};
    }
  };
  const request = await dispatchGuest(control, record, {version: 1, workflow: id, operation, action: "ready"});
  assert.equal(request.guestCommand?.phase, "submitted");
  let before = effects.length;
  const checked = (await reconcileGuest(control, request)).record;
  assert.deepEqual(effects.slice(before).map(x => x.split(":")[0]), ["GET", "persist"]);
  assert.equal(checked.phase, "provisioned");
  assert.equal(checked.guestCommand?.phase, "failed");
  assert.equal(checked.guestReady, undefined);
  assert.equal(checked.readinessReceipts, undefined);
  assert.equal(checked.assessment, undefined);
  assert.equal(checked.migration, undefined);
  before = effects.length;
  await assert.rejects(ensureAssessmentReadiness(control, checked), /provisioned runner/);
  for (const action of ["profile", "inventory"] as const) {
    await assert.rejects(dispatchGuest(control, checked, {version: 1, workflow: id, operation, action, configuration: {source: {type: "neo4j"}}}), /fresh guest readiness/);
    const reviewed = {...checked, sourceDraft: {canAssess: true, configuration: {source: {type: "neo4j"}}, warnings: [], form: {} as never}};
    await assert.rejects(startAssessment(control, reviewed, action, {}), /fresh guest readiness|complete inventory/);
  }
  assert.deepEqual(effects.slice(before), []);
  assert.equal(saved.some(r => r.assessment || r.guestReady || r.migration), false);
});
