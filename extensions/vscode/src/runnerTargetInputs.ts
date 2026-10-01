import * as vscode from "vscode";
import { RunnerRecord } from "./core/runner";
import { RunnerControl } from "./core/runnerLifecycle";
import { availableTargetSubnets, targetNetworkGroup, validateTargetSubnet } from "./core/runnerTarget";

export async function pickTargetSubnet(control:RunnerControl,record:RunnerRecord,previous?:string,reuse=false):Promise<string|undefined> {
  if(!vscode.workspace.isTrusted)return undefined;
  targetNetworkGroup(record);
  const vnetId=record.input.subnetId.replace(/\/subnets\/[^/]+$/i,"");
  const result=await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"Finding free PostgreSQL subnet ranges",cancellable:true},async(_progress,token)=>{
    const response=await control.request(record.input.subscriptionId,`${vnetId}?api-version=2024-05-01`);
    if(token.isCancellationRequested || !vscode.workspace.isTrusted)return undefined;
    if(response.status!==200)throw new Error("Cannot read the runner VNet address space. Check network read permissions and retry target review; no subnet was selected.");
    const choices=availableTargetSubnets(response.value);
    let previousError:string|undefined;
    if(previous){
      try { validateTargetSubnet(previous,response.value); }
      catch(error){previousError=error instanceof Error?error.message:String(error);}
      if(!previousError)choices.unshift(previous);
    }
    return {choices:[...new Set(choices)],previousError};
  });
  if(!result || !vscode.workspace.isTrusted)return undefined;
  if(!result.choices.length)throw new Error("No free IPv4 /28 range remains in the runner VNet. Ask the network owner to add address space, or start a new workflow in another VNet. Existing subnets will not be reused or changed.");
  if(reuse && previous && !result.previousError)return previous;
  const choice=await vscode.window.showQuickPick(result.choices.map(prefix=>({
    label:prefix,description:prefix===previous?"Saved target range":"New PostgreSQL subnet",
    detail:`VNet ${vnetId.split("/").at(-1)}; free range, not an existing subnet. Rechecked before deployment.`,
    prefix
  })),{title:"Select a new delegated PostgreSQL subnet",ignoreFocusOut:true,
    placeHolder:result.previousError?`Saved ${previous} cannot be reused: ${result.previousError} Select a free range.`:"Choose a free range; existing source and runner subnets are excluded."});
  return vscode.workspace.isTrusted?choice?.prefix:undefined;
}

export async function pickTargetDeadline(previous?:string,reuse=false):Promise<string|undefined> {
  if(!vscode.workspace.isTrusted)return undefined;
  const now=Date.now(),remaining=previous?Date.parse(previous)-now:NaN;
  if(reuse && remaining>0 && remaining<=96*3600000)return previous;
  const choices=[
    {label:"1 hour",hours:1},{label:"6 hours",hours:6},{label:"12 hours",hours:12},
    {label:"1 day",hours:24},{label:"2 days",hours:48},{label:"3 days",hours:72}
  ];
  const choice=await vscode.window.showQuickPick(choices,{
    title:"Cost authorization duration",ignoreFocusOut:true,
    placeHolder:reuse?"Saved deadline has expired or is invalid. Choose a new duration; no automatic renewal.":"Duration from selection time; UTC is calculated. This does not schedule shutdown."
  });
  if(!choice || !vscode.workspace.isTrusted)return undefined;
  return new Date(Date.now()+choice.hours*3600000).toISOString();
}
