import * as vscode from "vscode";
import { RunnerStore } from "./guided/runnerStore";

/** Native, local-only review; no Azure session or controller is constructed. */
export async function reviewRunnerCrashLock(store: RunnerStore): Promise<void> {
  const trusted = () => { if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before reviewing an interrupted runner lock."); };
  trusted();
  const workflows = await store.list();
  if (!workflows.length) { await vscode.window.showInformationMessage("No retained runner workflows are available for lock review."); return; }
  const selected = await vscode.window.showQuickPick(workflows.map(r => ({ label: r.id, description: r.input.resourceGroup, id: r.id })),
    { placeHolder: "Select a workflow to review its interrupted local lock" });
  if (!selected) return;
  trusted();
  const review = await store.reviewCrashLock(selected.id);
  const confirmation = "Recover local lock";
  const answer = await vscode.window.showWarningMessage("Recover this interrupted runner lock?", {
    modal: true,
    detail: `Workflow: ${review.workflowId}\nFormer local process: ${review.owner.pid}\nLock created: ${review.owner.createdAt}\nLock SHA-256: ${review.lockSHA256}\nWorkflow SHA-256: ${review.recordSHA256}\n\nThe owning process is absent in this OS boot session. Its lock evidence will be archived before removal. Workflow records and reports remain intact. A remote operation may still be running; this action does not reconnect, retry, resume, or submit any operation. Review / reconcile the retained operation separately.`
  }, confirmation);
  await store.recoverCrashLock(review, answer === confirmation, trusted);
  if (answer === confirmation) await vscode.window.showInformationMessage("Interrupted local lock recovered and archived. No remote action was sent. Review / reconcile the retained operation before approving further work.");
}
