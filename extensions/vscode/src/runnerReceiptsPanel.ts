import * as vscode from "vscode";
import { RunnerStore } from "./guided/runnerStore";
import { readinessArchive, readinessReceiptReferenced } from "./core/runnerReceipts";

/** Local-only evidence export. Available before target creation and while stopped. */
export async function archiveRunnerReadiness(store: RunnerStore): Promise<void> {
  const trusted = () => {
    if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before archiving runner evidence.");
  };
  trusted();
  const workflows = (await store.list()).filter(r => r.readinessReceipts?.length);
  if (!workflows.length) {
    await vscode.window.showInformationMessage("No sealed readiness receipts are available. Legacy ARM commands are not automatically adopted or removed.");
    return;
  }
  const selected = await vscode.window.showQuickPick(workflows.map(r => ({ label: r.id, description: r.input.resourceGroup, id: r.id })), { placeHolder: "Select runner evidence to archive locally (no Azure changes)" });
  if (!selected) return;
  trusted();
  const record = await store.read(selected.id);
  const choice = await vscode.window.showQuickPick((record.readinessReceipts ?? []).map(receipt => ({
    label: receipt.command.id.slice(receipt.command.id.lastIndexOf("/") + 1),
    description: `${receipt.command.submittedAt} — ${readinessReceiptReferenced(record, receipt) ? "still referenced; preserve" : "historical readiness only"}`,
    receipt
  })), { placeHolder: "Archive one successful readiness receipt; nothing will be deleted" });
  if (!choice) return;
  const expected = readinessArchive(record, choice.receipt);
  const text = await store.exclusive(record.id, async () => {
    trusted();
    const latest = await store.read(record.id);
    if (latest.vmId !== record.vmId || latest.input.subscriptionId !== record.input.subscriptionId) throw new Error("Workflow changed during archive review. Select it again.");
    const checked = readinessArchive(latest, choice.receipt);
    if (checked.text !== expected.text) throw new Error("Readiness receipt changed during archive review.");
    await store.retainReport(record.id, checked.manifest, checked.text);
    return store.readReport(record.id, checked.manifest);
  });
  const document = await vscode.workspace.openTextDocument({ language: "json", content: text });
  await vscode.window.showTextDocument(document, { preview: true });
  await vscode.window.showInformationMessage("Readiness receipt archived and hash-verified locally. No Azure command, VM, disk, source data, or migration evidence was deleted. This archive alone does not authorize command removal.");
}
