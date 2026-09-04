import * as assert from "node:assert/strict";
import * as vscode from "vscode";

suite("AGEFreighter extension", () => {
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
