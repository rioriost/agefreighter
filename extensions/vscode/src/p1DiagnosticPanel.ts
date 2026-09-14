import * as vscode from "vscode";
import {createHash,randomUUID} from "node:crypto";
import {open} from "node:fs/promises";
import {dirname,join} from "node:path";
import {RunnerControl} from "./core/runnerLifecycle";
import {RunnerStore} from "./guided/runnerStore";
import {AzureSession} from "./guided/azure";
import {object,RunnerRecord} from "./core/runner";
import {developmentEnabled} from "./developmentRunner";
import {developmentArtifact} from "./core/runnerDevelopment";
import {inspectCSV} from "./guided/csvTransfer";
import {verifyTransferStorage} from "./core/runnerReportStorage";
import {p1Root,p1FixtureRoot,parseP1Receipt} from "./core/p1Qualification";
import {diagnosticGate,diagnosticReceipt,p1DiagnosticScript,P1Diagnostic} from "./core/p1Diagnostic";
import {targetDSN} from "./core/runnerExecution";
import {escapeHTML} from "./core/report";

export async function diagnoseP1(context:vscode.ExtensionContext,control:RunnerControl,store:RunnerStore,azure:AzureSession,id:string):Promise<RunnerRecord>{
  if(!developmentEnabled()||!vscode.workspace.isTrusted)throw new Error("Trusted development workflow required.");
  let r=await store.read(id);
  if(r.p1Diagnostic){
    return store.exclusive(id,async()=>{
      r=await store.read(id);const d=r.p1Diagnostic!;
      if(d.jobId!==r.migration?.jobId||d.failedOperation!==r.p1Qualification?.operation||d.commandId!==`${r.vmId}/runCommands/af-${d.operation}`)throw new Error("Diagnostic identity changed.");
      if(d.phase==="failed")throw new Error("Diagnostic failed; preserve evidence, no automatic replay.");
      if(d.phase!=="finished"){
        const response=await control.request(r.input.subscriptionId,`${d.commandId}?api-version=2024-07-01&$expand=instanceView`),v=object(object(object(response.value).properties).instanceView);
        if(!["Succeeded","Failed","TimedOut","Canceled"].includes(String(v.executionState)))return r;
        try{if(v.executionState!=="Succeeded"||v.exitCode!==0)throw new Error();d.result=diagnosticReceipt(parseP1Receipt(v.output),r,d);d.phase="finished";}
        catch{d.phase="failed";await control.persist(r);throw new Error("Diagnostic receipt failed; preserve evidence, no replay.");}
        await control.persist(r);
      }
      const panel=vscode.window.createWebviewPanel("agefreighter.p1Diagnostic","P1 failure diagnosis",vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});
      panel.webview.html=`<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>P1 failure diagnosis — not qualification PASS</h1><pre>${escapeHTML(JSON.stringify(d.result,null,2))}</pre>`;
      return r;
    });
  }
  diagnosticGate(r);
  const selected=await vscode.window.showOpenDialog({canSelectMany:false,filters:{"P1 diagnostic verifier manifest":["json"]},openLabel:"Review diagnostic verifier"});
  if(!selected?.[0]||selected[0].scheme!=="file")return r;
  const file=await open(selected[0].fsPath,"r");let raw:Record<string,unknown>;
  try{const stat=await file.stat();if(!stat.isFile()||stat.size>16384)throw new Error("Invalid manifest");raw=object(JSON.parse(await file.readFile("utf8")));}finally{await file.close();}
  if(raw.purpose!=="p1-read-only-verifier"||raw.fixtureRoot!==p1FixtureRoot||raw.canonicalRoot!==p1Root||typeof raw.archive!=="string"||!/^[A-Za-z0-9_.-]+\.tar\.gz$/.test(raw.archive))throw new Error("Not the frozen P1 verifier manifest.");
  const artifact=developmentArtifact(r,raw),path=join(dirname(selected[0].fsPath),raw.archive),manifest=await inspectCSV(id,path);
  if(manifest.sha256!==artifact.sha256||manifest.bytes!==artifact.development!.bytes)throw new Error("Verifier archive changed.");
  if(await vscode.window.showWarningMessage("Run the approved read-only P1 failure diagnosis?",{modal:true,detail:`Job ${r.migration!.jobId}; retained failure ${r.p1Qualification!.operation}. Commit ${artifact.development!.commit}; SHA-256 ${artifact.sha256}. New separate operation; no migration replay, graph changes, loader replacement or deletion of old failure/active marker. Up to 4 GiB RAM, zero swap, 25 minutes. Uses the same target credential through protected transport. Fixed diagnostic codes only; never qualification PASS.`},"Run read-only diagnosis")!=="Run read-only diagnosis")return r;
  return store.exclusive(id,async()=>{
    r=await store.read(id);diagnosticGate(r);if(r.p1Diagnostic)throw new Error("Reconcile existing diagnostic.");
    const response=await control.request(r.input.subscriptionId,`${r.target!.serverId}?api-version=2024-08-01`),s=object(response.value),t=object(s.tags);
    if(response.status!==200||t.workflow!==id||t.application!=="agefreighter"||t.purpose!=="migration-target"||object(s.properties).state!=="Ready")throw new Error("Target ownership/readiness changed.");
    await verifyTransferStorage(control,r);
    if((await control.list(r.input.subscriptionId,`${r.vmId}/runCommands?api-version=2024-07-01`)).length>=25)throw new Error("Command capacity reached; preserve receipts.");
    await azure.uploadRunnerArchive(r,path,manifest);
    const operation=randomUUID(),d:P1Diagnostic={operation,commandId:`${r.vmId}/runCommands/af-${operation}`,jobId:r.migration!.jobId,failedOperation:r.p1Qualification!.operation,artifact,phase:"submitted",startedAt:new Date().toISOString()};
    if((await control.request(r.input.subscriptionId,`${d.commandId}?api-version=2024-07-01`)).status!==404)throw new Error("Command already exists.");
    const key=`runner-target/${id}/${createHash("sha256").update(r.target!.serverId).digest("hex")}`,password=await context.secrets.get(key);
    if(!password)throw new Error("Retained target credential unavailable.");
    const script=p1DiagnosticScript(r,d);r={...r,p1Diagnostic:d};await control.persist(r);
    try{const response=await control.request(r.input.subscriptionId,`${d.commandId}?api-version=2024-07-01`,"PUT",{location:r.input.region,properties:{source:{script},protectedParameters:[{name:"AF_P1_DSN",value:targetDSN(r,password)}],timeoutInSeconds:1800,asyncExecution:true}});if(response.status<200||response.status>=300)throw new Error();}
    catch{d.phase="unknown";await control.persist(r);}
    return r;
  });
}
