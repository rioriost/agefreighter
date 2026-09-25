import * as assert from "node:assert/strict";
import * as vscode from "vscode";
import { createHash, randomUUID } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { RunnerStore } from "../../guided/runnerStore";
import { AzureSession } from "../../guided/azure";
import { RunnerRecord, sourceWorkflowDraft } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { reportStorageNames } from "../../core/runnerReportStorage";
import { qualifyP1 } from "../../p1QualificationPanel";
import { P1RejectedImportError } from "../../core/p1Qualification";

// Real VS Code, production controller and on-disk store. Only ARM is an inert,
// GET-only fixture. Synthetic target envelopes are NOT new Azure qualification.
type Digest = {source: string; jobId?: string; recordCount: number; rootSha256: string;
  leaves: {kind: string; name: string; sha256: string}[]};
type Report = {version: number; jobId: string; readOnly: boolean; expected: Digest;
  actual: Digest; comparison: {status: string}};
const hash = (text: string) => createHash("sha256").update(text).digest("hex");
const cases: {name: string; error?: RegExp; change?: (d: Report) => void; corruptBytes?: boolean; truncate?: boolean}[] = [
  {name: "complete retained canonical report"},
  {name: "foreign envelope job", error: /identity or outcome/, change: d => {d.jobId = randomUUID();}},
  {name: "foreign target job", error: /identity or outcome/, change: d => {d.actual.jobId = randomUUID();}},
  {name: "missing range", error: /coverage/, change: d => {d.actual.leaves.pop();}},
  {name: "duplicated range", error: /root mismatch/, change: d => {d.actual.leaves[1] = {...d.actual.leaves[0]!};}},
  {name: "reordered ranges", error: /root mismatch/, change: d => {d.actual.leaves.reverse();}},
  {name: "forged matching roots and leaves", error: /root mismatch/, change: d => {
    d.expected.leaves[0]!.sha256 = "a".repeat(64); d.actual.leaves[0]!.sha256 = "a".repeat(64);
  }},
  {name: "wrong record count", error: /coverage/, change: d => {d.actual.recordCount--;}},
  {name: "failed comparison", error: /identity or outcome/, change: d => {d.comparison.status = "fail";}},
  {name: "non-read-only result", error: /identity or outcome/, change: d => {d.readOnly = false;}},
  {name: "truncated retained JSON", error: /JSON|Unexpected|Expected/, truncate: true},
  {name: "changed retained bytes", error: /SHA-256/, corruptBytes: true}
];

suite("P1 retained evidence in isolated real Extension Host", () => {
  let root: string;
  let previousOptIn: boolean | undefined;
  const results: object[] = [];
  suiteSetup(async () => {
    root = process.env.AF_ISOLATED_HOST_ROOT!;
    assert.ok(root && isAbsolute(root) && /^af-extension-host-/.test(root.split("/").pop()!));
    assert.equal(vscode.extensions.getExtension("rioriost.agefreighter")?.extensionPath, resolve(__dirname, "../../.."));
    assert.equal(vscode.workspace.isTrusted, true);
    previousOptIn = vscode.workspace.getConfiguration("agefreighter").inspect<boolean>("allowDevelopmentRunnerArtifacts")?.globalValue;
    // This disposable profile has no Azure account, installed operator store or
    // SecretStorage. Never enable development artifacts in the user's profile.
    await vscode.workspace.getConfiguration("agefreighter").update("allowDevelopmentRunnerArtifacts", true, vscode.ConfigurationTarget.Global);
  });
  suiteTeardown(async () => {
    await vscode.workspace.getConfiguration("agefreighter").update("allowDevelopmentRunnerArtifacts", previousOptIn, vscode.ConfigurationTarget.Global);
    const text = JSON.stringify({scope: "isolated real host; synthetic retained reports; inert ARM", vscode: vscode.version, results}, null, 2);
    await writeFile(join(root, "p1-retained-results.json"), text, {flag: "wx", mode: 0o600});
    console.log(`P1 retained host evidence SHA256 ${hash(text)}`);
  });

  for (const scenario of cases) test(scenario.name, async () => {
    await vscode.commands.executeCommand("workbench.action.closeAllEditors");
    const id = randomUUID(), storeRoot = join(root, `p1-${id}`), store = new RunnerStore(storeRoot);
    const expected: Digest = JSON.parse(await readFile(resolve(__dirname, "../../../../../production-simulation/vscode-e2e/evidence/p1-canonical-expected-20260906.json"), "utf8"));
    const doc: Report = {version: 1, jobId: id, readOnly: true, expected,
      actual: {...structuredClone(expected), source: "apache-age", jobId: id}, comparison: {status: "pass"}};
    scenario.change?.(doc);
    const text = scenario.truncate ? JSON.stringify(doc).slice(0, -1) : JSON.stringify(doc);
    const record = sourceWorkflowDraft(id, {subscriptionId: id, resourceGroup: "isolated-test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: {type: "csv", location: "local"}});
    record.migration = {jobId: id, phase: "finished", verification: {outcome: "pass", summary: "synthetic counts only"}} as RunnerRecord["migration"];
    record.p1Qualification = {operation: id, jobId: id, commandId: `${record.vmId}/runCommands/af-${id}`,
      phase: "pass", startedAt: new Date().toISOString(), artifact: record.artifact, sha256: hash(text), bytes: Buffer.byteLength(text)};
    await store.write(record);
    const recordPath = join(storeRoot, `${id}.json`), reportPath = join(storeRoot, `${id}.report-${id}.json`);
    // Deliberately seed an invalid retained file in the disposable test store;
    // normal retainReport correctly refuses malformed JSON before publication.
    if (scenario.truncate) await writeFile(reportPath, text, {flag: "wx", mode: 0o600});
    else await store.retainReport(id, {operation: id, sha256: hash(text), bytes: Buffer.byteLength(text)}, text);
    if (scenario.corruptBytes) await writeFile(reportPath, text.replace('"readOnly":true', '"readOnly":null'));
    const beforeRecord = await readFile(recordPath), beforeReport = await readFile(reportPath);
    const names = reportStorageNames(record), requests: string[] = [];
    let persists = 0;
    const control: RunnerControl = {
      sleep: async () => {}, list: async () => {throw Error("No ARM list allowed");},
      persist: async r => {persists++; await store.write(r);},
      request: async (_sub, path, method = "GET") => {
        assert.equal(method, "GET", "No cloud mutation is possible in this fixture"); requests.push(path);
        if (path === `${names.id}?api-version=2023-05-01`) return {status: 200, value: {
          id: names.id, location: "japaneast", tags: {application: "agefreighter", workflow: id, purpose: "artifact-transfer"},
          properties: {provisioningState: "Succeeded", supportsHttpsTrafficOnly: true, allowBlobPublicAccess: false,
            allowSharedKeyAccess: false, minimumTlsVersion: "TLS1_2", primaryEndpoints: {blob: `${names.origin}/`}}
        }};
        assert.equal(path, `${names.containerId}?api-version=2023-05-01`);
        return {status: 200, value: {id: names.containerId, properties: {publicAccess: "None"}}};
      }
    };
    // Trap all credentials/capabilities/uploads/downloads, including accidental
    // fallback to Azure after retained-file corruption. No AzureSession exists.
    const forbidden = new Proxy({}, {get: () => {throw Error("No credentials or network adapter allowed");}});
    const execute = () => qualifyP1(forbidden as vscode.ExtensionContext, control, store, forbidden as AzureSession, id);
    let rejectionCategory: string | undefined;
    if (scenario.error) {
      await assert.rejects(execute(), error => {
        assert.ok(error instanceof Error);assert.match(error.message, scenario.error!);
        if (!scenario.truncate && !scenario.corruptBytes) {
          assert.ok(error instanceof P1RejectedImportError);rejectionCategory = error.category;
          assert.deepEqual(error.evidence, {status: "rejected", retained: true,
            manifest: {operation: id, sha256: hash(text), bytes: Buffer.byteLength(text)}, jobId: id, profile: "raw-id"});
          assert.match(error.message, /Rejected evidence is retained; it is not an accepted P1 result/);
        }
        return true;
      });
      assert.equal(persists, 0);
      assert.deepEqual(await readFile(recordPath), beforeRecord);
      assert.deepEqual(await readFile(reportPath), beforeReport);
    } else {
      await execute(); assert.equal(persists, 1);
      for (let i = 0; i < 20 && !vscode.window.tabGroups.all.some(g => g.tabs.some(t => t.label === "Verified P1 migration")); i++) await new Promise(r => setTimeout(r, 50));
    }
    const panels = vscode.window.tabGroups.all.flatMap(g => g.tabs).filter(t => t.label === "Verified P1 migration");
    assert.equal(panels.length, scenario.error ? 0 : 1, "Only a valid canonical report may create the actual PASS tab");
    assert.equal(requests.length, 2, "Only inert storage ownership GETs; no export, download or replay");
    results.push({scenario: scenario.name, outcome: "pass", passTabs: panels.length, persists,
      realCloudRequests: 0, inertStorageGETs: requests.length, recordPreservedOnRejection: !!scenario.error,
      ...(rejectionCategory ? {rejectionCategory, retainedEvidenceStatus: "rejected"} : {})});
    await vscode.commands.executeCommand("workbench.action.closeAllEditors");
  });
});
