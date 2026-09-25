import * as vscode from "vscode";
import { VerificationDecision } from "./core/runnerVerification";
import { migrationVerificationView } from "./core/runnerVerificationView";

export function showMigrationVerification(decision: VerificationDecision, report: string): vscode.WebviewPanel {
  const content = migrationVerificationView(decision, report);
  const panel = vscode.window.createWebviewPanel("agefreighter.verifiedMigration", content.title,
    vscode.ViewColumn.Beside, {enableScripts: false, localResourceRoots: []});
  panel.webview.html = content.html;
  return panel;
}
