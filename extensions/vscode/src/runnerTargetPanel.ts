import * as vscode from "vscode";
import { randomBytes, createHash } from "node:crypto";
import { open } from "node:fs/promises";
import { join } from "node:path";
import { RunnerStore } from "./guided/runnerStore";
import { AzureSession } from "./guided/azure";
import { RunnerControl } from "./core/runnerLifecycle";
import { sourceTargetEvidence, targetPreview, TargetInput, submitTarget, repairBusyTargetPreload, targetStorageSizing, targetPending, targetStatusMessage } from "./core/runnerTarget";
import { RunnerRecord } from "./core/runner";
import { preflightTarget, targetComputeRate } from "./core/runnerTargetPreflight";
import { ensureAssessmentReadiness } from "./core/runnerAssessment";
import { targetDraftBinding, retainTargetDraft } from "./core/targetDraft";
import { pickTargetDeadline, pickTargetSubnet } from "./runnerTargetInputs";
import { watchTargetState } from "./runnerWatch";

export interface TargetReviewFeedback {
  cancelled?: () => boolean;
  progress?: (record: RunnerRecord, text: string, active: boolean) => Promise<void>;
}

export async function reviewRunnerTarget(context:vscode.ExtensionContext,control:RunnerControl,store:RunnerStore,azure:AzureSession,workflow?:string,feedback:TargetReviewFeedback={}):Promise<void>{
  if(!vscode.workspace.isTrusted)throw new Error("Trust this workspace before planning Azure resources.");
  const selected=workflow?{id:workflow}:await vscode.window.showQuickPick((await store.list()).filter(r=>r.phase==="provisioned" && ["csv","neo4j","postgresql","cosmos-nosql"].includes(r.input.source.type)).map(r=>({label:r.id,description:`${r.input.source.type} — ${r.input.resourceGroup} — ${r.target?.phase??"assess first"}`,id:r.id})),{placeHolder:"Review a complete source inventory and its private PostgreSQL target"});
  if(!selected)return;
  let record=await store.read(selected.id);
  let monitoringStopped=false;
  const reconcile=async()=>{
    await feedback.progress?.(record,targetStatusMessage(record),true);
    const result=await watchTargetState(control,store,record.id,feedback.cancelled??(()=>false),async current=>{
      record=current;
      await feedback.progress?.(current,targetStatusMessage(current),targetPending(current));
    });
    monitoringStopped=result===undefined;
    record=result??await store.read(record.id);
    const text=targetStatusMessage(record)+(targetPending(record)
      ? ` Monitoring ${result?"reached its time limit":"stopped"}; Azure work was not cancelled. Use Review / reconcile private target to resume monitoring.`
      : "");
    await feedback.progress?.(record,text,false);
    return text;
  };
  if(record.target && record.target.phase!=="previewed"){
    const text=await reconcile();
    if(feedback.cancelled?.() || !vscode.workspace.isTrusted)return;
    if(monitoringStopped){void vscode.window.showInformationMessage(text);return;}
    if(record.target?.phase==="failed" && !record.target.configurationRepair){
      // Read-only eligibility checks precede the narrow native approval.
      await repairBusyTargetPreload(control,record);
      const repair=await vscode.window.showWarningMessage("Repair only the failed AGE preload setting?",{modal:true,detail:`${record.target.serverId}\nThe original deployment failed only for shared_preload_libraries with ServerIsBusy; other resources succeeded. Apply the already reviewed pg_stat_statements,age value once, only if the server is Ready and the value remains its unchanged system default. Preserve the original failed deployment, server, database, credentials and evidence. No target recreation, automatic retry or migration. A later explicit restart may be needed. Deadline ${record.target.input.deadline}; ceiling USD ${record.target.input.budgetUSD}.`},"Repair reviewed preload only");
      if(repair==="Repair reviewed preload only")record=await store.exclusive(record.id,async()=>repairBusyTargetPreload(control,await store.read(record.id),true));
      await feedback.progress?.(record,targetStatusMessage(record),false);
      void vscode.window.showInformationMessage(`Preload repair: ${record.target?.configurationRepair?.phase??"not submitted"}. Reopen target review for read-only reconciliation; no deployment is replayed.`);return;
    }
    void vscode.window.showInformationMessage(text);return;
  }
  const a=record.assessment;
  if(!a?.reportSHA256 || !a.reportBytes)throw new Error("Complete and import the whole-source inventory first.");
  const report=await store.readReport(record.id,{operation:a.operation,sha256:a.reportSHA256,bytes:a.reportBytes});
  const evidence=sourceTargetEvidence(record,report);
  const binding=targetDraftBinding(record);
  const draft=record.targetDraft?.binding===binding?record.targetDraft:undefined;
  const values:Partial<TargetInput>={...record.target?.input,...draft?.input};
  let folderPath=draft?.folder;
  const save=async()=>{record=await store.exclusive(record.id,async()=>{const next=retainTargetDraft(await store.read(record.id),binding,values,folderPath);await store.write(next);return next;});};
  const required:(keyof TargetInput)[]=["serverName","subnetCIDR","postgresSKU","storageGiB","loaderSize","deadline","budgetUSD","additionalReserveUSD"];
  const reuse=required.every(k=>values[k]!==undefined)?await vscode.window.showQuickPick(["Reuse saved target inputs","Edit saved target inputs"],{placeHolder:`Saved inputs only — no deployment authorized. Deadline ${values.deadline}`}):"Edit saved target inputs";
  if(!reuse)return;
  const ask=async(key:keyof TargetInput,prompt:string,value:string)=>{
    if(reuse==="Reuse saved target inputs")return String(values[key]);
    const result=await vscode.window.showInputBox({prompt,value:values[key]===undefined?value:String(values[key]),ignoreFocusOut:true});
    if(result!==undefined){Object.assign(values,{[key]:["budgetUSD","additionalReserveUSD"].includes(key)?Number(result):result});await save();}return result;
  };
  const pick=async(key:keyof TargetInput,options:string[],placeHolder:string)=>{
    const previous=values[key]===undefined?undefined:String(values[key]);
    if(reuse==="Reuse saved target inputs" && previous && options.includes(previous))return previous;
    const sorted=previous&&options.includes(previous)?[previous,...options.filter(x=>x!==previous)]:options;
    const result=(await vscode.window.showQuickPick(sorted.map(value=>({label:key==="storageGiB"?`${value} GiB`:value,value})),{placeHolder,ignoreFocusOut:true}))?.value;
    if(result!==undefined){Object.assign(values,{[key]:key==="storageGiB"?Number(result):result});await save();}return result;
  };
  const serverName=await ask("serverName","New private PostgreSQL 18 server name",`afpg-${record.id.replaceAll("-","").slice(0,20)}`);if(serverName===undefined)return;
  const subnetCIDR=await pickTargetSubnet(control,record,values.subnetCIDR,reuse==="Reuse saved target inputs");if(subnetCIDR===undefined)return;
  values.subnetCIDR=subnetCIDR;await save();
  const postgresSKU=await pick("postgresSKU",["Standard_D4ds_v5","Standard_D8ds_v5","Standard_D16ds_v5","Standard_E8ds_v5"],"Target SKU (not a throughput guarantee)");if(!postgresSKU)return;
  const sizingStorage=targetStorageSizing(evidence.storageHighBytes);
  if(!sizingStorage.options.length)throw new Error(`Target storage needs ${sizingStorage.requiredGiB} GiB including 25% headroom; the supported choices stop at 1024 GiB. No target was deployed.`);
  const storage=await pick("storageGiB",sizingStorage.options.map(String),`Target storage: high estimate ${sizingStorage.highGiB} GiB; with 25% headroom ${sizingStorage.requiredGiB} GiB (rounded up)`);if(!storage)return;
  const loaderSize=await pick("loaderSize",["Standard_D4s_v5","Standard_D8s_v5","Standard_D16s_v5"],"Same Linux VM's migration size; 4 GiB loader RSS remains the bound");if(!loaderSize)return;
  const deadline=await pickTargetDeadline(values.deadline,reuse==="Reuse saved target inputs");if(deadline===undefined)return;
  values.deadline=deadline;await save();
  const budget=await ask("budgetUSD","Approved cumulative workflow cost ceiling, USD","100");if(budget===undefined)return;
  const reserve=await ask("additionalReserveUSD","Reserve USD including accrued charges, delayed billing and retained storage/network","50");if(reserve===undefined)return;
  const input:TargetInput={serverName,subnetCIDR,postgresSKU,postgresTier:postgresSKU.startsWith("Standard_E")?"MemoryOptimized":"GeneralPurpose",storageGiB:Number(storage),loaderSize,deadline,budgetUSD:Number(budget),additionalReserveUSD:Number(reserve),hourlyUSD:1};
  input.hourlyUSD=targetComputeRate(await azure.retailRates(record.input.region,[loaderSize,postgresSKU]),input);
  Object.assign(values,input);await save();
  const plan=targetPreview({...record,target:undefined},input,evidence);
  const sizing=evidence.sourceType==="neo4j"?"Neo4j sizing high bound uses exact count-store totals at 16 KiB per mapped record; this is conservative for P1 but not a universal property-width guarantee.":evidence.sourceType==="postgresql"?`${Object.keys(evidence.labels).length} mapped labels were counted in one complete repeatable-read stream.`:evidence.sourceType==="cosmos-nosql"?`${Object.keys(evidence.labels).length} mapped labels were counted in one complete stream under the reviewed source-immutability window.`:`${Object.keys(evidence.labels).length} mapped labels were counted by the complete CSV scan.`;
  const choice=await vscode.window.showWarningMessage("Review the private migration target and same-VM sizing",{modal:true,detail:
    `${evidence.rows} mapped rows; inventory SHA-256 ${evidence.reportSHA256}. ${sizing}\nMigration group: ${record.input.resourceGroup}, ${record.input.region}, zone ${record.input.zone}\nNew delegated subnet: ${plan.subnetId}\n${plan.networkDeployment?`Network group: ${plan.networkDeployment.resourceGroup}. Separate child deployment: ${plan.networkDeployment.deploymentId}. Both scopes require deployment permission; only the new subnet is declared in the network group.\n`:""}PostgreSQL 18 / AGE: ${postgresSKU}, ${storage} GiB; HA disabled (single-server trial). ${subnetCIDR}, private DNS in the migration group linked to the existing VNet. No public access or peering.\nSame runner: ${loaderSize}; resize is a later, separate idle-VM operation.\nCompute USD ${input.hourlyUSD}/hour + USD ${input.additionalReserveUSD} accrued/non-compute reserve. Total ceiling USD ${input.budgetUSD}; deadline ${deadline}. This is a budget gate, not a guaranteed bill or automatic shutdown.\nFolder selection saves a secret-reference-only LoadJob. Generated target credentials stay only in VS Code SecretStorage. Deployment is followed by separate AGE readiness, migration and full verification; it does not mark completion.`},"Save plan and approve target deployment","Save plan only");
  if(!choice)return;
  if(!folderPath){
    const folder=await vscode.window.showOpenDialog({canSelectFiles:false,canSelectFolders:true,canSelectMany:false,openLabel:"Save reviewed LoadJob and target plan here"});
    if(!folder?.[0] || folder[0].scheme!=="file")return;
    folderPath=folder[0].fsPath;await save();
  }
  const checkTarget=async(current:typeof record,refreshReadiness:boolean)=>vscode.window.withProgress(
    {location:vscode.ProgressLocation.Notification,title:"Checking private target readiness",cancellable:true},
    async(progress,token)=>{
      const cancelled=()=>token.isCancellationRequested||feedback.cancelled?.()===true||!vscode.workspace.isTrusted;
      if(refreshReadiness){
        progress.report({message:"Checking Linux readiness; no source reads."});
        current=await ensureAssessmentReadiness(control,current,cancelled);
      }
      await preflightTarget(control,current,input,{cancelled,progress:message=>progress.report({message})});
      return current;
    });
  await store.exclusive(record.id,async()=>{
    let latest=await store.read(record.id);
    if(latest.target && latest.target.phase!=="previewed" || JSON.stringify(sourceTargetEvidence(latest,report))!==JSON.stringify(evidence))throw new Error("Workflow changed while reviewing; no deployment was submitted.");
    // JSON is a strict YAML 1.2 subset. This export is directly accepted by the CLI,
    // uses guest paths and environment references, and never includes a password.
    if(!vscode.workspace.isTrusted || targetDraftBinding(latest)!==binding)throw new Error("Target review changed or trust was revoked.");
    if(choice==="Save plan and approve target deployment"){
      latest=await checkTarget(latest,true);
    }
    const currentPlan=targetPreview({...latest,target:undefined},input,evidence);
    const stem=`agefreighter-${record.id}-${currentPlan.hash.slice(0,12)}-${Date.now()}`;
    for(const [name,data] of [[`${stem}.yaml`,JSON.stringify(latest.sourceDraft!.configuration,null,2)+"\n"],[`${stem}.target.json`,JSON.stringify(currentPlan,null,2)+"\n"]]){
      const f=await open(join(folderPath!,name!),"wx",0o600);try{await f.writeFile(data!);await f.sync();}finally{await f.close();}
    }
    const next={...latest,target:currentPlan};await control.persist(next);
    if(choice!=="Save plan and approve target deployment")return;
    const secretKey=`runner-target/${record.id}/${createHash("sha256").update(plan.serverId).digest("hex")}`;
    let password=await context.secrets.get(secretKey);
    if(!password){password=randomBytes(32).toString("base64url")+"Aa1!";await context.secrets.store(secretKey,password);}
    await submitTarget(control,next,password,async()=>{
      await checkTarget(next,false);
      const live=targetComputeRate(await azure.retailRates(record.input.region,[loaderSize,postgresSKU]),input);
      if(live!==input.hourlyUSD)throw new Error("Compute price changed; review a new plan before deployment.");
    });
  });
  record=await store.read(record.id);
  const text=choice==="Save plan only"?"Reviewed LoadJob and plan saved. No Azure resources were deployed.":await reconcile();
  if(choice==="Save plan only")await feedback.progress?.(record,text,false);
  if(record.target?.phase==="failed" && !monitoringStopped)throw new Error(targetStatusMessage(record));
  if(!feedback.cancelled?.() && vscode.workspace.isTrusted)void vscode.window.showInformationMessage(text);
}
