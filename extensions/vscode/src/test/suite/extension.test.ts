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
});
