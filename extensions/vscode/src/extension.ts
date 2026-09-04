import * as vscode from "vscode";
import { registerAI } from "./ai";
import { ExtensionController } from "./controller";
import { registerRunnerMigration } from "./runnerMigration";
import { JobsProvider } from "./jobsView";

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel("AGEFreighter", { log: true });
  const jobs = new JobsProvider();
  const controller = new ExtensionController(jobs, output);

  context.subscriptions.push(
    output,
    jobs,
    vscode.window.registerTreeDataProvider("agefreighter.jobs", jobs)
  );
  controller.register(context);
  registerRunnerMigration(context);
  registerAI(context, controller, jobs);
  output.appendLine("AGEFreighter extension activated.");
}

export function deactivate(): void {
  // Resources are registered in ExtensionContext.subscriptions.
}
