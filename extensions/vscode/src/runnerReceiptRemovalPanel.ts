import * as vscode from "vscode";
import { AzureSession } from "./guided/azure";
import { RunnerStore } from "./guided/runnerStore";
import { RunnerControl } from "./core/runnerLifecycle";
import { previewReceiptRemoval, reconcileReceiptRemoval, RemovalIO, submitReceiptRemoval } from "./core/runnerReceiptRemoval";

/** No batch cleanup or automatic retry. One explicitly selected control at a time. */
export async function manageReadinessRemoval(control: RunnerControl, store: RunnerStore, azure: AzureSession): Promise<void> {
  const trust = () => { if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before reviewing command removal."); };
  trust();
  const workflows = (await store.list()).filter(r => r.readinessReceipts?.length);
  if (!workflows.length) {
    await vscode.window.showInformationMessage("No sealed readiness receipts are available. Legacy ARM commands are not automatically adopted or removed.");
    return;
  }
  const selected = await vscode.window.showQuickPick(workflows.map(r => ({ label: r.id, description: r.input.resourceGroup, id: r.id })), { placeHolder: "Select retained runner readiness evidence" });
  if (!selected) return;
  const record = await store.read(selected.id);
  const choice = await vscode.window.showQuickPick((record.readinessReceipts ?? []).map(receipt => {
    const intent = record.readinessRemovals?.find(x => x.commandId === receipt.command.id);
    return { label: receipt.command.id.slice(receipt.command.id.lastIndexOf("/") + 1), description: intent ? `GET-only reconciliation: ${intent.phase}` : "Review one old readiness control (requires running VM and fresh idle readiness)", receipt, intent };
  }), { placeHolder: "No VM start/stop or automatic command replay" });
  if (!choice) return;
  trust();
  const binding = choice.intent?.accountBinding ?? await azure.runnerAccountBinding(record.input.subscriptionId);
  const io: RemovalIO = {
    control: { ...control, persist: async r => { await control.persist(r); await store.syncEvidenceDirectory(); } },
    guard: async expected => { trust(); if (await azure.runnerAccountBinding(record.input.subscriptionId) !== expected) throw new Error("Azure account changed; review again."); trust(); },
    retain: async (manifest, text) => { await store.retainReport(record.id, manifest, text); await store.syncEvidenceDirectory(); },
    read: manifest => store.readReport(record.id, manifest),
    remove: (commandId, expected) => azure.removeRunnerReadiness(record.input.subscriptionId, commandId, expected)
  };
  if (choice.intent) {
    const updated = await store.exclusive(record.id, async () => {
      const latest = await store.read(record.id);
      if (latest.vmId !== record.vmId || latest.input.subscriptionId !== record.input.subscriptionId) throw new Error("Workflow scope changed.");
      return reconcileReceiptRemoval(io, latest, choice.receipt.command.id);
    });
    const phase = updated.readinessRemovals?.find(x => x.commandId === choice.receipt.command.id)?.phase;
    await vscode.window.showInformationMessage(phase === "absent" ? "Azure control record absence verified. Local archive and guest evidence remain; the Azure record itself is not recoverable by this action." : "Removal is not confirmed. Evidence retained; no DELETE was retried. Operator review may be required.");
    return;
  }
  const plan = await previewReceiptRemoval(io, record, choice.receipt.command.id, binding);
  if (await vscode.window.showWarningMessage("Archive and permanently remove this one Azure readiness control record?", { modal: true,
    detail: `${plan.commandId}\nReceipt SHA-256: ${plan.receiptSHA256}\nOnly this successful, unreferenced readiness record is removed. The owned VM must already be running, with a separate newer same-boot readiness control proving fresh idle health. Both controls must still match their sealed successful evidence. Latest readiness is preserved. A verified local archive of both proofs and a single-use intent are saved first. Disks, guest evidence, jobs and source data are preserved; the Azure control record cannot be restored by this action. Coordinate exclusive control: do not modify the VM or submit work from another client during this step. Pending or changed evidence blocks removal. Lost replies are reconciled by GET only, never an automatic DELETE retry. This does not start compute or a migration or extend an approved runtime window.` }, "Archive and remove this record") !== "Archive and remove this record") return;
  await store.exclusive(record.id, async () => submitReceiptRemoval(io, await store.read(record.id), plan, true));
  await vscode.window.showInformationMessage("Removal intent retained. Select this record again to verify absence by GET. No deletion-complete claim is made until absence is confirmed.");
}
