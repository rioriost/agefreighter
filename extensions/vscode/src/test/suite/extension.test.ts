import * as assert from "node:assert/strict";
import * as vscode from "vscode";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { createHash } from "node:crypto";
import { RunnerStore } from "../../guided/runnerStore";
import { sourceWorkflowDraft, SourceKind } from "../../core/runner";
import { openRunnerSource } from "../../runnerSourcePanel";
import { assessCountsVerification, VerificationExpectation } from "../../core/runnerVerification";
import { showMigrationVerification } from "../../migrationVerificationPanel";

const verificationExpected: VerificationExpectation = {
  jobId: "11111111-1111-4111-8111-111111111111", fingerprint: "a".repeat(64),
  cliVersion: "2.4.0", startedAt: "2026-09-05T00:00:00Z", labels: {"v.Person": 2}
};
function completeReport() {
  return {
    schemaVersion: 1, command: "verify", agefreighterVersion: "2.4.0",
    generatedAt: "2026-09-05T00:01:00Z", outcome: "pass",
    job: {id: verificationExpected.jobId, configFingerprint: verificationExpected.fingerprint},
    checks: ["job-status", "graph-generation", "generation-ownership"].map(id => ({id, status: "pass"})),
    errors: [], warnings: [], incompleteChecks: [] as string[],
    sections: [{title: "Per-label counts", fields: [
      {name: "v.Person", status: "pass", value: "acceptedRows=2,committedRows=2,livePhysicalRows=2,liveIdentityRows=2,rejectedRows=0,counterCompleteness=complete,storedPhysicalComparison=verified,physicalIdentityEquality=verified"},
      {name: "unclassified.rejects", status: "pass", value: "0"}
    ]}]
  };
}

suite("AGEFreighter extension", () => {
  // Actual VS Code panels; synthetic evidence only. No signed-in store,
  // cloud control, secret access or changes to accepted qualification records.
  for (const scenario of [
    {name: "complete counts", outcome: "pass", change: (_d: ReturnType<typeof completeReport>) => {}},
    {name: "wrong job", outcome: "incomplete", change: (d: ReturnType<typeof completeReport>) => {d.job.id = randomUUID();}},
    {name: "stale evidence", outcome: "incomplete", change: (d: ReturnType<typeof completeReport>) => {d.generatedAt = "2026-09-04T23:59:00Z";}},
    {name: "missing counts", outcome: "incomplete", change: (d: ReturnType<typeof completeReport>) => {d.sections = [];}},
    {name: "incomplete coverage", outcome: "incomplete", change: (d: ReturnType<typeof completeReport>) => {d.incompleteChecks = ["bounded-integrity"];}},
    {name: "count mismatch", outcome: "fail", change: (d: ReturnType<typeof completeReport>) => {d.sections[0]!.fields[0]!.value = d.sections[0]!.fields[0]!.value.replace("livePhysicalRows=2", "livePhysicalRows=1");}},
    {name: "rejected records", outcome: "fail", change: (d: ReturnType<typeof completeReport>) => {d.sections[0]!.fields[1]!.value = "1";}},
    {name: "failed integrity check", outcome: "fail", change: (d: ReturnType<typeof completeReport>) => {d.checks[0]!.status = "fail";}},
    {name: "truncated document", outcome: "incomplete", truncate: true, change: (_d: ReturnType<typeof completeReport>) => {}},
    {name: "hash mismatch", outcome: "incomplete", wrongHash: true, change: (_d: ReturnType<typeof completeReport>) => {}}
  ]) {
    test(`verification panel: ${scenario.name}`, async () => {
      const doc = completeReport(); scenario.change(doc);
      const reportJSON = scenario.truncate ? JSON.stringify(doc).slice(0, -1) : JSON.stringify(doc);
      const decision = assessCountsVerification(verificationExpected, {exitCode: 0, reportJSON,
        sha256: scenario.wrongHash ? "0".repeat(64) : createHash("sha256").update(reportJSON).digest("hex")}, Date.parse("2026-09-05T00:05:00Z"));
      assert.equal(decision.outcome, scenario.outcome);
      const panel = showMigrationVerification(decision, reportJSON);
      try {
        assert.equal(panel.webview.options.enableScripts, false);
        assert.deepEqual(panel.webview.options.localResourceRoots, []);
        if (scenario.outcome === "pass") {
          assert.equal(panel.title, "AGEFreighter counts verified");
          assert.match(panel.webview.html, /Full property-digest qualification remains a separate check/);
        } else {
          assert.doesNotMatch(panel.title, /verified/i);
          assert.match(panel.webview.html, /<h1>Verification: (FAILED|INCOMPLETE) — migration is not complete<\/h1>/);
        }
        for (let attempt = 0; attempt < 20 && !vscode.window.tabGroups.all.some(group => group.tabs.some(tab => tab.label === panel.title)); attempt++) await new Promise(resolve => setTimeout(resolve, 50));
        assert.ok(vscode.window.tabGroups.all.some(group => group.tabs.some(tab => tab.label === panel.title)));
      } finally {panel.dispose();}
    });
  }
  test("opens all four local source editors without a release, workspace, CLI or ARM request", async () => {
    const store = new RunnerStore(await mkdtemp(join(tmpdir(), "af-source-host-")));
    const subscriptions: vscode.Disposable[] = [];
    let requests = 0;
    for (const type of ["neo4j", "postgresql", "cosmos-nosql", "csv"] as SourceKind[]) {
      const id = randomUUID(), record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type, location: type === "csv" ? "local" : "azure" } });
      await store.write(record);
      openRunnerSource({ subscriptions } as vscode.ExtensionContext, { sleep: async () => {}, persist: r => store.write(r), list: async () => { requests++; throw new Error("Unexpected ARM list"); }, request: async () => { requests++; throw new Error("Unexpected ARM request"); } }, store, id);
      for (let attempt = 0; attempt < 20 && !vscode.window.tabGroups.all.some(group => group.tabs.some(tab => tab.label === "AGEFreighter source assessment")); attempt++) await new Promise(resolve => setTimeout(resolve, 50));
      assert.ok(vscode.window.tabGroups.all.some(group => group.tabs.some(tab => tab.label === "AGEFreighter source assessment")));
      await vscode.commands.executeCommand("workbench.action.closeAllEditors");
    }
    assert.equal(requests, 0);
    subscriptions.forEach(disposable => disposable.dispose());
  });
  test("activates and registers deterministic commands", async () => {
    const extension = vscode.extensions.getExtension("rioriost.agefreighter");
    assert.ok(extension, "extension is installed in the Extension Host");
    await extension.activate();
    assert.equal(extension.isActive, true);

    const commands = await vscode.commands.getCommands(true);
    for (const command of [
      "agefreighter.newGuidedMigration",
      "agefreighter.validate",
      "agefreighter.plan",
      "agefreighter.profile",
      "agefreighter.doctor",
      "agefreighter.load",
      "agefreighter.resume",
      "agefreighter.status",
      "agefreighter.verify",
      "agefreighter.report",
      "agefreighter.optimize",
      "agefreighter.cleanup"
    ]) {
      assert.ok(commands.includes(command), `${command} is registered`);
    }
  });

  test("opens runner-first wizard without a workspace or local CLI selection", async () => {
    assert.equal(vscode.workspace.workspaceFolders?.length ?? 0, 0);
    let deadline: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        vscode.commands.executeCommand("agefreighter.newGuidedMigration"),
        new Promise((_, reject) => { deadline = setTimeout(() => reject(new Error("Wizard unexpectedly waits for folder/CLI selection")), 2000); })
      ]);
      // Panel creation crosses the extension-host boundary; wait for the UI's
      // tab inventory to acknowledge it rather than inspecting the prior tick.
      for (let attempt = 0; attempt < 20 && !vscode.window.tabGroups.all.some(group => group.tabs.some(tab => tab.label === "New AGEFreighter migration")); attempt++) {
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      const tabs = vscode.window.tabGroups.all.flatMap(group => group.tabs);
      assert.ok(tabs.some(tab => tab.label === "New AGEFreighter migration"));
    } finally {
      if (deadline) clearTimeout(deadline);
      await vscode.commands.executeCommand("workbench.action.closeAllEditors");
    }
  });
});
