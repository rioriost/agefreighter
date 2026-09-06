import * as vscode from "vscode";
import { randomBytes, createHash } from "node:crypto";
import { open } from "node:fs/promises";
import { join } from "node:path";
import { RunnerStore } from "./guided/runnerStore";
import { AzureSession } from "./guided/azure";
import { RunnerControl } from "./core/runnerLifecycle";
import { csvTargetEvidence, targetPreview, TargetInput, submitTarget, refreshTarget } from "./core/runnerTarget";
import { preflightTarget, targetComputeRate } from "./core/runnerTargetPreflight";

export async function reviewRunnerTarget(context:vscode.ExtensionContext,control:RunnerControl,store:RunnerStore,azure:AzureSession,workflow?:string):Promise<void>{
  if(!vscode.workspace.isTrusted)throw new Error("Trust this workspace before planning Azure resources.");
  const selected=workflow?{id:workflow}:await vscode.window.showQuickPick((await store.list()).filter(r=>r.phase==="provisioned" && r.input.source.type==="csv").map(r=>({label:r.id,description:`${r.input.resourceGroup} — ${r.target?.phase??"assess first"}`,id:r.id})),{placeHolder:"Review a complete CSV inventory and its private PostgreSQL target"});
  if(!selected)return;
  let record=await store.read(selected.id);
  if(record.target && record.target.phase!=="previewed"){
    record=await store.exclusive(record.id,async()=>refreshTarget(control,await store.read(record.id)));
    await vscode.window.showInformationMessage(`Private target: ${record.target?.phase}. This is ARM status, not AGE readiness, migration or verification. No operation was replayed.`);return;
  }
  const a=record.assessment;
  if(!a?.reportSHA256 || !a.reportBytes)throw new Error("Complete and import the whole-source CSV inventory first.");
  const report=await store.readReport(record.id,{operation:a.operation,sha256:a.reportSHA256,bytes:a.reportBytes});
  const evidence=csvTargetEvidence(record,report);
  const ask=(prompt:string,value:string)=>vscode.window.showInputBox({prompt,value,ignoreFocusOut:true});
  const serverName=await ask("New private PostgreSQL 18 server name",record.target?.input.serverName??`afpg-${record.id.replaceAll("-","").slice(0,20)}`);if(serverName===undefined)return;
  const subnetCIDR=await ask("New non-overlapping delegated subnet CIDR inside the existing runner VNet",record.target?.input.subnetCIDR??"");if(subnetCIDR===undefined)return;
  const postgresSKU=await vscode.window.showQuickPick(["Standard_D4ds_v5","Standard_D8ds_v5","Standard_D16ds_v5","Standard_E8ds_v5"],{placeHolder:"Target SKU (4-vCore GP is a starting point for bounded trials, not a throughput guarantee)"});if(!postgresSKU)return;
  const storage=await vscode.window.showQuickPick(["128","256","512","1024"],{placeHolder:`Target GiB: must cover high estimate ${evidence.storageHighBytes} bytes plus 25% headroom`});if(!storage)return;
  const loaderSize=await vscode.window.showQuickPick(["Standard_D4s_v5","Standard_D8s_v5","Standard_D16s_v5"],{placeHolder:"Same Linux VM's migration size; 4 GiB loader RSS remains the bound"});if(!loaderSize)return;
  const deadline=await ask("Approved UTC live-window deadline (ISO 8601; never extends an existing authorization)",record.target?.input.deadline??new Date(Date.now()+24*3600000).toISOString());if(deadline===undefined)return;
  const budget=await ask("Approved total workflow cost ceiling, USD",String(record.target?.input.budgetUSD??100));if(budget===undefined)return;
  const reserve=await ask("Reserve USD covering accrued charges, delayed billing, all storage/NAT/network/backup and retained evidence through the deadline",String(record.target?.input.additionalReserveUSD??50));if(reserve===undefined)return;
  const input:TargetInput={serverName,subnetCIDR,postgresSKU,postgresTier:postgresSKU.startsWith("Standard_E")?"MemoryOptimized":"GeneralPurpose",storageGiB:Number(storage),loaderSize,deadline,budgetUSD:Number(budget),additionalReserveUSD:Number(reserve),hourlyUSD:1};
  input.hourlyUSD=targetComputeRate(await azure.retailRates(record.input.region,[loaderSize,postgresSKU]),input);
  await preflightTarget(control,record,input);
  const plan=targetPreview({...record,target:undefined},input,evidence);
  const choice=await vscode.window.showWarningMessage("Review the private CSV migration target and same-VM sizing",{modal:true,detail:
    `${evidence.rows} mapped rows across ${Object.keys(evidence.labels).length} labels; inventory SHA-256 ${evidence.reportSHA256}\n${record.input.resourceGroup}, ${record.input.region}, zone ${record.input.zone}\nPostgreSQL 18 / AGE: ${postgresSKU}, ${storage} GiB; HA disabled (single-server trial). ${subnetCIDR}, private DNS in the existing VNet. No public access or peering.\nSame runner: ${loaderSize}; resize is a later, separate idle-VM operation.\nCompute USD ${input.hourlyUSD}/hour + USD ${input.additionalReserveUSD} accrued/non-compute reserve. Total ceiling USD ${input.budgetUSD}; deadline ${deadline}. This is a budget gate, not a guaranteed bill or automatic shutdown.\nFolder selection saves a secret-reference-only LoadJob. A generated administrator password is stored only in VS Code SecretStorage. Deployment is followed by separate AGE readiness, migration and full verification; it does not mark completion.`},"Save plan and approve target deployment","Save plan only");
  if(!choice)return;
  const folder=await vscode.window.showOpenDialog({canSelectFiles:false,canSelectFolders:true,canSelectMany:false,openLabel:"Save reviewed LoadJob and target plan here"});
  if(!folder?.[0] || folder[0].scheme!=="file")return;
  await store.exclusive(record.id,async()=>{
    const latest=await store.read(record.id);
    if(latest.target && latest.target.phase!=="previewed" || JSON.stringify(csvTargetEvidence(latest,report))!==JSON.stringify(evidence))throw new Error("Workflow changed while reviewing; no deployment was submitted.");
    // JSON is a strict YAML 1.2 subset. This export is directly accepted by the CLI,
    // uses guest paths and environment references, and never includes a password.
    const stem=`agefreighter-${record.id}-${plan.hash.slice(0,12)}`;
    for(const [name,data] of [[`${stem}.yaml`,JSON.stringify(latest.sourceDraft!.configuration,null,2)+"\n"],[`${stem}.target.json`,JSON.stringify(plan,null,2)+"\n"]]){
      const f=await open(join(folder[0]!.fsPath,name!),"wx",0o600);try{await f.writeFile(data!);await f.sync();}finally{await f.close();}
    }
    const next={...latest,target:plan};await control.persist(next);
    if(choice!=="Save plan and approve target deployment")return;
    const secretKey=`runner-target/${record.id}/${createHash("sha256").update(plan.serverId).digest("hex")}`;
    let password=await context.secrets.get(secretKey);
    if(!password){password=randomBytes(32).toString("base64url")+"Aa1!";await context.secrets.store(secretKey,password);}
    await submitTarget(control,next,password,async()=>{
      await preflightTarget(control,next,input);
      const live=targetComputeRate(await azure.retailRates(record.input.region,[loaderSize,postgresSKU]),input);
      if(live!==input.hourlyUSD)throw new Error("Compute price changed; review a new plan before deployment.");
    });
  });
  await vscode.window.showInformationMessage(choice==="Save plan only"?"Reviewed LoadJob and plan saved. No Azure resources were deployed.":"Target intent retained. Reopen target review to reconcile ARM status; do not replay deployment. AGE readiness and migration remain separate.");
}
