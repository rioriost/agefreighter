import * as assert from "node:assert/strict";
import * as vscode from "vscode";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { RunnerStore } from "../../guided/runnerStore";
import { sourceWorkflowDraft, SourceKind } from "../../core/runner";
import { openRunnerSource } from "../../runnerSourcePanel";

suite("AGEFreighter extension", () => {
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
