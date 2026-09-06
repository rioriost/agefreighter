import * as vscode from "vscode";
import {createHash} from "node:crypto";
import {RunnerControl} from "./core/runnerLifecycle";
import {RunnerStore} from "./guided/runnerStore";
import {AzureSession} from "./guided/azure";
import {startResize,advanceResize} from "./core/runnerResize";
import {migrationPreflight,startMigration,refreshMigration,verifyMigrationReport,applyTargetPreload} from "./core/runnerExecution";
import {startReportExport,refreshReportExport,importReport} from "./core/runnerReport";
import {escapeHTML} from "./core/report";
import {targetComputeRate} from "./core/runnerTargetPreflight";
import {diagnoseTarget,archiveEmptyTargetFailure} from "./core/runnerDiagnostic";
import {qualifyP1} from "./p1QualificationPanel";

/** Native choices are intentionally separate approvals. Reconnecting or closing
 * a panel cannot launch/resume a migration, resize, or repeat a lost operation. */
export async function continueRunnerExecution(context:vscode.ExtensionContext,control:RunnerControl,store:RunnerStore,azure:AzureSession,workflow?:string):Promise<void>{
  if(!vscode.workspace.isTrusted)throw new Error("Trust this workspace before controlling migration resources.");
  const selected=workflow?{id:workflow}:await vscode.window.showQuickPick((await store.list()).filter(r=>r.target?.phase==="provisioned").map(r=>({label:r.id,description:`${r.input.resourceGroup} — ${r.migration?.phase??r.resize?.phase??"resize required"}`,id:r.id})),{placeHolder:"Select the retained private CSV target"});
  if(!selected)return;
  let r=await store.read(selected.id);
  const action=await vscode.window.showQuickPick(["Apply / reconcile AGE preload restart","Reconcile resize (read only)","Approve next same-VM resize step","Start new CSV migration and counts verification","Refresh retained migration (never replay)","Transfer / open migration verification","Diagnose retained CSV target (read only)","Archive empty-target preparation failure","Qualify / reconcile full P1 digest (development only)"],{placeHolder:`Runner: ${r.resize?.phase??"not resized"}; migration: ${r.migration?.phase??"not started"}`});
  if(!action)return;
  const confirm=(title:string,detail:string)=>vscode.window.showWarningMessage(title,{modal:true,detail},"Approve this step");
  const price=async()=>{if(!r.target)throw new Error("No retained target plan.");const input=r.target.input;if(targetComputeRate(await azure.retailRates(r.input.region,[input.loaderSize,input.postgresSKU]),input)!==input.hourlyUSD)throw new Error("Compute price changed; review the cost plan before further mutation.");};
  if(action==="Apply / reconcile AGE preload restart"){
    const approved=!!r.targetRestart || await confirm("Apply the AGE preload configuration?",`Restart only ${r.target?.serverId} if its approved preload parameter requires it. This is before migration. Existing data is retained. An already submitted restart is reconciled by read only.`)==="Approve this step";
    r=await store.exclusive(r.id,async()=>applyTargetPreload(control,await store.read(r.id),approved));
  }else if(action==="Reconcile resize (read only)"){
    r=await store.exclusive(r.id,async()=>advanceResize(control,await store.read(r.id)));
  }else if(action==="Approve next same-VM resize step"){
    if(!r.target)throw new Error("Review the private target first.");
    if(await confirm("Resize the existing idle Linux runner?",`${r.vmId}\n${r.resize?.phase??"Deallocate before resizing"} → ${r.target.input.loaderSize}. The same NIC, disk and system identity are preserved. No source VM is changed. Data and evidence remain. Deadline ${r.target.input.deadline}; total ceiling USD ${r.target.input.budgetUSD}. An uncertain response is reconciled, never replayed.`)!=="Approve this step")return;
    await price();
    r=await store.exclusive(r.id,async()=>{const latest=await store.read(r.id);return latest.resize?advanceResize(control,latest,true):startResize(control,latest);});
  }else if(action==="Start new CSV migration and counts verification"){
    const a=r.assessment;if(!a?.reportSHA256 || !a.reportBytes)throw new Error("Import complete CSV inventory first.");
    const report=await store.readReport(r.id,{operation:a.operation,sha256:a.reportSHA256,bytes:a.reportBytes});
    const evidence=await migrationPreflight(control,r,report);
    if(await confirm("Start this new CSV migration on the Linux runner?",`${evidence.rows} mapped rows / ${Object.keys(evidence.labels).length} labels. Inventory ${evidence.reportSHA256}. Linux ${r.artifact.version}, archive ${r.artifact.sha256}.\n${r.target!.serverId}\nPrepare AGE, create the new graph, migrate, then run complete counts verification. The new job UUID is retained before writes. No replace, delete, automatic resume or retry. Private target credentials stay in SecretStorage and protected transport. A counts pass is distinct from the independent P1 property digest.`)!=="Approve this step")return;
    await price();
    r=await store.exclusive(r.id,async()=>{
      const latest=await store.read(r.id);
      const key=`runner-target/${r.id}/${createHash("sha256").update(latest.target!.serverId).digest("hex")}`,password=await context.secrets.get(key);
      if(!password)throw new Error("The retained target credential is unavailable.");
      return startMigration(control,latest,report,password);
    });
  }else if(action==="Refresh retained migration (never replay)"){
    r=await store.exclusive(r.id,async()=>refreshMigration(control,await store.read(r.id)));
  }else if(action==="Qualify / reconcile full P1 digest (development only)"){
    r=await qualifyP1(context,control,store,azure,r.id);
  }else if(action==="Archive empty-target preparation failure"){
    if(await confirm("Archive this preparation failure without deleting or resuming anything?",`Requires fresh read-only proof that the target graph and metadata schema are absent. Retains the failed job and diagnostic in history and all guest evidence. This permits a separately approved runner repair and a new create-only job, not replay or replacement of existing data.`)!=="Approve this step")return;
    r=await store.exclusive(r.id,async()=>archiveEmptyTargetFailure(control,await store.read(r.id)));
  }else if(action==="Diagnose retained CSV target (read only)"){
    if(!r.targetDiagnostic && await confirm("Read-only diagnosis of the retained CSV target?",`Run the pinned Linux CLI doctor without --persist against the existing target. No load, resume, AGE preparation or resource changes. Private credentials remain in SecretStorage/protected transport. Raw redacted evidence stays on the guest.`)!=="Approve this step")return;
    r=await store.exclusive(r.id,async()=>{
      const latest=await store.read(r.id);
      const key=`runner-target/${r.id}/${createHash("sha256").update(latest.target!.serverId).digest("hex")}`;
      return diagnoseTarget(control,latest,latest.targetDiagnostic?undefined:await context.secrets.get(key));
    });
    if(r.targetDiagnostic?.result){
      const view=vscode.window.createWebviewPanel("agefreighter.targetDiagnostic","AGEFreighter target diagnostic",vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});
      view.webview.html=`<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>Read-only target diagnostic</h1><pre>${escapeHTML(JSON.stringify(r.targetDiagnostic.result,null,2))}</pre>`;
    }
  }else{
    const m=r.migration;if(!m?.reportSHA256 || !m.reportBytes)throw new Error("Refresh the terminal migration verification manifest first.");
    const manifest={operation:m.operation,sha256:m.reportSHA256,bytes:m.reportBytes};
    if(await confirm("Transfer and verify the retained migration report?",`${m.jobId}\n${manifest.bytes} bytes; SHA-256 ${manifest.sha256}. Retained privately on this Mac, not sent to an AI model. This never reruns migration.`)!=="Approve this step")return;
    r=await store.exclusive(r.id,async()=>{
      let latest=await store.read(r.id);
      if(latest.migration?.operation!==m.operation || latest.migration.reportSHA256!==manifest.sha256)throw new Error("Migration report changed during review.");
      if(!latest.reportTransfers?.some(t=>t.operation===m.operation))return startReportExport(control,latest,await azure.reportCapability(latest,m.operation,"c"),m.operation);
      if(latest.guestCommand?.action==="export-report" && latest.guestCommand.operation===m.operation)latest=await refreshReportExport(control,latest);
      if(latest.reportTransfers?.find(t=>t.operation===m.operation)?.phase!=="imported")latest=await importReport(control,latest,m.operation,await azure.reportCapability(latest,m.operation,"r"),(id,m,text)=>store.retainReport(id,m,text));
      const text=await store.readReport(latest.id,manifest),next=verifyMigrationReport(latest,text);await control.persist(next);return next;
    });
    if(r.migration?.verification){
      const text=await store.readReport(r.id,manifest),view=vscode.window.createWebviewPanel("agefreighter.verifiedMigration","Verified AGEFreighter migration",vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});
      view.webview.html=`<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>${escapeHTML(r.migration.verification.summary)}</h1><pre>${escapeHTML(text)}</pre>`;
    }
  }
  void vscode.window.showInformationMessage(`Target preload: ${r.targetRestart?.phase??"not checked"}; runner resize: ${r.resize?.phase??"not started"}; migration: ${r.migration?.phase??"not started"}; counts: ${r.migration?.verification?.outcome??"not verified"}. Independent P1 property digest is a separate qualification gate.`);
}
