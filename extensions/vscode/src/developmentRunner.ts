import * as vscode from "vscode";
import { dirname, join } from "node:path";
import { open } from "node:fs/promises";
import { AzureSession } from "./guided/azure";
import { RunnerStore } from "./guided/runnerStore";
import { RunnerControl } from "./core/runnerLifecycle";
import { developmentArtifact } from "./core/runnerDevelopment";
import { inspectCSV } from "./guided/csvTransfer";
import { object } from "./core/runner";
import { verifyTransferStorage } from "./core/runnerReportStorage";
import { assertUpgradeIdle, refreshUpgrade, submitUpgrade } from "./core/runnerUpgrade";

export function developmentEnabled(): boolean {
  // A repository/workspace setting cannot authorize executable development code.
  return vscode.workspace.getConfiguration("agefreighter").inspect<boolean>("allowDevelopmentRunnerArtifacts")?.globalValue === true;
}

export async function upgradeDevelopmentRunner(control: RunnerControl, store: RunnerStore, azure: AzureSession): Promise<void> {
  if (!developmentEnabled() || !vscode.workspace.isTrusted) throw new Error("A trusted workspace and user-level development artifact opt-in are required.");
  const picked = await vscode.window.showQuickPick((await store.list()).filter(r => r.phase === "provisioned" && r.artifact.development).map(record => ({label:record.id,description:`${record.input.resourceGroup} — ${record.upgrade?.phase ?? record.artifact.version}`,record})), {placeHolder:"Upgrade or reconcile a retained development runner (preserve all CSVs and evidence)"});
  if (!picked) return;
  if (picked.record.upgrade && picked.record.upgrade.phase !== "finished") {
    const next = await store.exclusive(picked.record.id, async () => refreshUpgrade(control, await store.read(picked.record.id)));
    await vscode.window.showInformationMessage(`Runner upgrade: ${next.upgrade?.phase}. ${next.upgrade?.phase === "finished" ? "Refresh guest readiness before assessment." : "No installation was replayed. Failed/uncertain installation requires retained evidence review."}`);
    return;
  }
  assertUpgradeIdle(picked.record);
  const selected = await vscode.window.showOpenDialog({canSelectMany:false,filters:{"Pinned development manifest":["json"]},openLabel:"Review replacement Linux manifest"});
  if (!selected?.[0] || selected[0].scheme !== "file") return;
  const f = await open(selected[0].fsPath,"r"); let raw:Record<string,unknown>;
  try {
    const info=await f.stat(); if(!info.isFile() || info.size>16384) throw new Error("Manifest must be a bounded regular file.");
    const b=Buffer.alloc(16385), result=await f.read(b,0,b.length,0);
    if(result.bytesRead>16384)throw new Error("Manifest exceeded its bound.");
    raw=object(JSON.parse(b.subarray(0,result.bytesRead).toString("utf8")));
  } finally {await f.close();}
  if(typeof raw.archive!=="string" || !/^[A-Za-z0-9_.-]+\.tar\.gz$/.test(raw.archive))throw new Error("Archive must be a same-directory tar.gz filename.");
  const artifact=developmentArtifact(picked.record,raw), path=join(dirname(selected[0].fsPath),raw.archive), manifest=await inspectCSV(picked.record.id,path);
  if(manifest.sha256!==artifact.sha256 || manifest.bytes!==artifact.development!.bytes)throw new Error("Archive differs from its reviewed manifest.");
  const approved=await vscode.window.showWarningMessage("Upgrade this idle Linux runner, preserving its data and previous installation?",{modal:true,
    detail:`VM: ${picked.record.vmId}\nOld: ${picked.record.artifact.version}\nNew: ${artifact.version}\nCommit: ${artifact.development!.commit}\nSHA-256: ${artifact.sha256}\nThe existing CSVs, seals, reports and old binaries remain. No automatic retry, rollback, migration or resource creation. Partial installation blocks further guest commands until reviewed. This unpublished artifact is only for qualification.`},"Approve pinned runner upgrade");
  if(approved!=="Approve pinned runner upgrade")return;
  await store.exclusive(picked.record.id,async()=>{
    const record=await store.read(picked.record.id);assertUpgradeIdle(record);
    if(record.artifact.sha256!==picked.record.artifact.sha256)throw new Error("Installed artifact changed; review again.");
    await verifyTransferStorage(control,record);
    await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"Uploading reviewed runner upgrade",cancellable:false},async()=>azure.uploadRunnerArchive(record,path,manifest));
    await submitUpgrade(control,record,artifact);
  });
  await vscode.window.showInformationMessage("Upgrade submitted once. Run this command again to reconcile its result; then refresh guest readiness.");
}

export async function prepareDevelopmentRunner(control: RunnerControl, store: RunnerStore, azure: AzureSession): Promise<void> {
  if (!developmentEnabled() || !vscode.workspace.isTrusted) throw new Error("Development artifacts require an explicit user-level opt-in and a trusted workspace. Production uses the matching published release.");
  const picked = await vscode.window.showQuickPick((await store.list()).filter(r => r.phase === "draft" && r.storageDeployment?.phase === "ready").map(record => ({ label: record.id, description: `${record.input.resourceGroup} / ${record.input.region}`, record })), { placeHolder: "Select a local draft with prepared transfer storage" });
  if (!picked) return;
  const selected = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { "Pinned development manifest": ["json"] }, openLabel: "Review Linux development archive manifest" });
  if (!selected?.[0] || selected[0].scheme !== "file") return;
  const f = await open(selected[0].fsPath, "r"); let raw: Record<string,unknown>;
  try {
    const info = await f.stat(); if (!info.isFile() || info.size > 16384) throw new Error("Development manifest must be a small regular JSON file.");
    const buffer = Buffer.alloc(16385), result = await f.read(buffer,0,buffer.length,0);
    if (result.bytesRead > 16384) throw new Error("Development manifest exceeded its bound.");
    raw = object(JSON.parse(buffer.subarray(0,result.bytesRead).toString("utf8")));
  } finally { await f.close(); }
  if (typeof raw.archive !== "string" || !/^[A-Za-z0-9_.-]+\.tar\.gz$/.test(raw.archive)) throw new Error("Archive must be a same-directory tar.gz filename.");
  const artifact = developmentArtifact(picked.record, raw), path = join(dirname(selected[0].fsPath), raw.archive);
  const manifest = await inspectCSV(picked.record.id, path);
  if (manifest.sha256 !== artifact.sha256 || manifest.bytes !== artifact.development!.bytes) throw new Error("Development archive differs from its pinned manifest.");
  const approved = await vscode.window.showWarningMessage("Prepare this unpublished executable for an isolated qualification runner?", { modal:true,
    detail:`Version: ${artifact.version}\nCommit: ${artifact.development!.commit}\nSHA-256: ${artifact.sha256}\n${manifest.bytes} bytes\n${artifact.url}\nThe manifest asserts build provenance; approve only your reviewed build. A later VM approval grants its identity Blob Reader on this workflow container. No Marketplace/GitHub release or source migration is performed. Production release verification is unchanged.` }, "Approve pinned test artifact");
  if (approved !== "Approve pinned test artifact") return;
  await store.exclusive(picked.record.id, async () => {
    let record = await store.read(picked.record.id);
    if (record.phase !== "draft" || JSON.stringify(record.input) !== JSON.stringify(picked.record.input)) throw new Error("Runner placement changed; review the development artifact again.");
    if (record.developmentUpload && JSON.stringify(record.developmentUpload.artifact) !== JSON.stringify(artifact)) throw new Error("A different pinned artifact is retained; use a new workflow.");
    await verifyTransferStorage(control,record);
    record = {...record,developmentUpload:{artifact,phase:"prepared"}};await control.persist(record);
    await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"Uploading reviewed Linux development archive",cancellable:false}, async()=>azure.uploadRunnerArchive(record,path,manifest));
    await control.persist({...record,artifact,developmentUpload:{artifact,phase:"ready"}});
  });
  await vscode.window.showInformationMessage("Pinned development archive is prepared. Reconnect to the draft, review the VM preview and its scoped Blob Reader grant.");
}
