import * as vscode from "vscode";
import {createHash,randomUUID} from "node:crypto";
import {open} from "node:fs/promises";
import {dirname,join} from "node:path";
import {developmentEnabled} from "./developmentRunner";
import {RunnerStore} from "./guided/runnerStore";
import {AzureSession} from "./guided/azure";
import {RunnerControl} from "./core/runnerLifecycle";
import {object,RunnerRecord} from "./core/runner";
import {assertIdleHealth} from "./core/runnerGuest";
import {targetBudget} from "./core/runnerTarget";
import {targetDSN} from "./core/runnerExecution";
import {developmentArtifact} from "./core/runnerDevelopment";
import {inspectCSV} from "./guided/csvTransfer";
import {verifyTransferStorage} from "./core/runnerReportStorage";
import {downloadReport,reportCapability,reportManifest} from "./core/runnerBlob";
import {p1Root,p1FixtureRoot,p1Script,p1ExportScript,verifyP1,P1Qualification} from "./core/p1Qualification";
import {escapeHTML} from "./core/report";

export async function qualifyP1(context:vscode.ExtensionContext,control:RunnerControl,store:RunnerStore,azure:AzureSession,id:string):Promise<RunnerRecord>{
  if(!developmentEnabled()||!vscode.workspace.isTrusted)throw new Error("P1 qualification requires user-level development opt-in and a trusted workspace.");
  let r=await store.read(id);
  if(r.migration?.phase!=="finished"||r.migration.verification?.outcome!=="pass")throw new Error("Import passing complete counts verification first.");
  if(r.p1Qualification){
    return store.exclusive(id,async()=>{
      r=await store.read(id);let q=r.p1Qualification!;
      if(q.jobId!==r.migration!.jobId || q.commandId!==`${r.vmId}/runCommands/af-${q.operation}`)throw new Error("Qualification job identity changed.");
      if(["submitted","unknown"].includes(q.phase)){
        const response=await control.request(r.input.subscriptionId,`${q.commandId}?api-version=2024-07-01&$expand=instanceView`),view=object(object(object(response.value).properties).instanceView);
        if(!["Succeeded","Failed","TimedOut","Canceled"].includes(String(view.executionState)))return r;
        let result:unknown;try{result=JSON.parse(String(view.output));}catch{}
        const v=object(result);
        if(view.executionState!=="Succeeded"||view.exitCode!==0||v.workflow!==id||v.operation!==q.operation||v.jobId!==q.jobId||v.verified!==true){r={...r,p1Qualification:{...q,phase:"failed"}};await control.persist(r);throw new Error("P1 qualification failed. Retain evidence; do not replay.");}
        reportManifest({operation:q.operation,sha256:String(v.sha256),bytes:Number(v.bytes)});
        q={...q,phase:"verified",sha256:String(v.sha256),bytes:Number(v.bytes)};r={...r,p1Qualification:q};await control.persist(r);
      }
      if(q.phase==="failed")throw new Error("Review retained qualification failure; no automatic retry.");
      if(q.phase==="verified"){
        targetBudget(r.target!.input);await verifyTransferStorage(control,r);
        if((await control.list(r.input.subscriptionId,`${r.vmId}/runCommands?api-version=2024-07-01`)).length>=25)throw new Error("Archive completed ARM receipts before result export.");
        const capability=await azure.reportCapability(r,q.operation,"c");reportCapability(capability,id,q.operation,"c");
        const exportCommandId=`${r.vmId}/runCommands/af-${randomUUID()}`;
        q={...q,phase:"exporting",exportCommandId};r={...r,p1Qualification:q};await control.persist(r);
        try{await control.request(r.input.subscriptionId,`${exportCommandId}?api-version=2024-07-01`,"PUT",{location:r.input.region,properties:{source:{script:p1ExportScript(r,q)},protectedParameters:[{name:"AF_P1_REPORT",value:capability}],timeoutInSeconds:300,asyncExecution:true}});}catch{/* Reconcile this exact command; never replay verification or export. */}
        return r;
      }
      if(q.phase==="exporting"){
        if(!q.exportCommandId?.startsWith(`${r.vmId}/runCommands/af-`))throw new Error("Invalid export identity.");
        const response=await control.request(r.input.subscriptionId,`${q.exportCommandId}?api-version=2024-07-01&$expand=instanceView`),view=object(object(object(response.value).properties).instanceView);
        if(!["Succeeded","Failed","TimedOut","Canceled"].includes(String(view.executionState)))return r;
        let value:unknown;try{value=JSON.parse(String(view.output));}catch{}
        const v=object(value);
        if(view.executionState!=="Succeeded"||view.exitCode!==0||v.workflow!==id||v.operation!==q.operation||v.jobId!==q.jobId||v.exported!==true||v.sha256!==q.sha256||v.bytes!==q.bytes){r={...r,p1Qualification:{...q,phase:"failed"}};await control.persist(r);throw new Error("Result export failed; verification evidence is retained. Do not replay.");}
        q={...q,phase:"exported"};r={...r,p1Qualification:q};await control.persist(r);
      }
      const manifest=reportManifest({operation:q.operation,sha256:q.sha256!,bytes:q.bytes!});
      await verifyTransferStorage(control,r);
      const text=q.phase==="pass"?await store.readReport(id,manifest):await downloadReport(await azure.reportCapability(r,q.operation,"r"),id,manifest);
      await store.retainReport(id,manifest,text);verifyP1(text,q.jobId);
      r={...r,p1Qualification:{...q,phase:"pass"}};await control.persist(r);
      const panel=vscode.window.createWebviewPanel("agefreighter.p1Qualification","Verified P1 migration",vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});
      panel.webview.html=`<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>P1 full canonical digest: PASS</h1><p>1,600,000 vertices and 4,000,000 edges. All 64 ranges, typed properties, identities and endpoints agree.</p><pre>${escapeHTML(text)}</pre>`;
      return r;
    });
  }
  assertIdleHealth(r);targetBudget(r.target!.input);
  const selected=await vscode.window.showOpenDialog({canSelectMany:false,filters:{"P1 verifier manifest":["json"]},openLabel:"Review pinned P1 verifier"});
  if(!selected?.[0]||selected[0].scheme!=="file")return r;
  const file=await open(selected[0].fsPath,"r");let raw:Record<string,unknown>;
  try{const stat=await file.stat();if(!stat.isFile()||stat.size>16384)throw new Error("Invalid manifest size.");raw=object(JSON.parse(await file.readFile("utf8")));}finally{await file.close();}
  if(raw.purpose!=="p1-read-only-verifier"||raw.fixtureRoot!==p1FixtureRoot||raw.canonicalRoot!==p1Root||typeof raw.archive!=="string"||!/^[A-Za-z0-9_.-]+\.tar\.gz$/.test(raw.archive))throw new Error("Not the frozen P1 verifier manifest.");
  const artifact=developmentArtifact(r,raw),path=join(dirname(selected[0].fsPath),raw.archive),manifest=await inspectCSV(id,path);
  if(manifest.sha256!==artifact.sha256||manifest.bytes!==artifact.development!.bytes)throw new Error("Verifier archive changed.");
  if(await vscode.window.showWarningMessage("Run independent full P1 verification on the existing Linux VM?",{modal:true,detail:`Read-only target job ${r.migration.jobId}. Regenerate the exact frozen P1 fixture; compare all 5.6M records and 64 canonical ranges. Commit ${artifact.development!.commit}, archive ${artifact.sha256}. This isolated verifier does not change the installed loader, graph, credentials or networking. Uses up to 4 GiB RAM and approximately 1 GiB retained fixture space; 25-minute execution cap. Results return privately to this Mac via the existing storage.`},"Approve full P1 verification")!=="Approve full P1 verification")return r;
  return store.exclusive(id,async()=>{
    const latest=await store.read(id);assertIdleHealth(latest);targetBudget(latest.target!.input);
    if(latest.p1Qualification||latest.migration?.jobId!==r.migration!.jobId||latest.migration.verification?.outcome!=="pass"||latest.guestCommand&&["submitted","unknown"].includes(latest.guestCommand.phase))throw new Error("Qualification state changed; reconcile before submission.");
    const s=await control.request(latest.input.subscriptionId,`${latest.target!.serverId}?api-version=2024-08-01`),v=object(s.value),t=object(v.tags);
    if(s.status!==200||t.workflow!==id||t.application!=="agefreighter"||!["migration-target","csv-migration-target"].includes(String(t.purpose))||object(v.properties).state!=="Ready")throw new Error("Target ownership/readiness changed.");
    await verifyTransferStorage(control,latest);
    if((await control.list(latest.input.subscriptionId,`${latest.vmId}/runCommands?api-version=2024-07-01`)).length>=25)throw new Error("Archive completed ARM receipts before qualification.");
    await azure.uploadRunnerArchive(latest,path,manifest);
    const operation=randomUUID(),q:P1Qualification={operation,commandId:`${latest.vmId}/runCommands/af-${operation}`,jobId:latest.migration.jobId,artifact,startedAt:new Date().toISOString(),phase:"submitted"};
    if((await control.request(latest.input.subscriptionId,`${q.commandId}?api-version=2024-07-01`)).status!==404)throw new Error("Qualification command already exists.");
    const key=`runner-target/${id}/${createHash("sha256").update(latest.target!.serverId).digest("hex")}`,password=await context.secrets.get(key);
    if(!password)throw new Error("Retained target credentials unavailable.");
    const script=p1Script(latest,q),next={...latest,p1Qualification:q};await control.persist(next);
    try{const response=await control.request(latest.input.subscriptionId,`${q.commandId}?api-version=2024-07-01`,"PUT",{location:latest.input.region,properties:{source:{script},protectedParameters:[{name:"AF_P1_DSN",value:targetDSN(latest,password)}],timeoutInSeconds:1800,asyncExecution:true}});if(response.status<200||response.status>=300)throw new Error();}
    catch{next.p1Qualification.phase="unknown";await control.persist(next);}
    return next;
  });
}
