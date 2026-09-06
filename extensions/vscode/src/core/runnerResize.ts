import {createHash} from "node:crypto";
import {object,RunnerRecord} from "./runner";
import {RunnerControl,preflightRunner} from "./runnerLifecycle";
import {assertIdleHealth} from "./runnerGuest";
import {targetBudget} from "./runnerTarget";

export interface RunnerResize {
  phase:"deallocating"|"ready-to-resize"|"resizing"|"ready-to-start"|"starting"|"finished";
  size:string; previousSize:string; preservedSHA256:string; startedAt:string; unknown?:boolean;
}
const digest=(x:unknown)=>createHash("sha256").update(JSON.stringify(x)).digest("hex");
async function inspect(control:RunnerControl,r:RunnerRecord){
  const response=await control.request(r.input.subscriptionId,`${r.vmId}?api-version=2024-07-01&$expand=instanceView`),v=object(response.value),p=object(v.properties),tags=object(v.tags);
  if(response.status!==200 || tags.workflow!==r.id || tags.application!=="agefreighter" || tags.purpose!=="discovery-and-migration" || v.location!==r.input.region || JSON.stringify(v.zones)!==JSON.stringify([r.input.zone]))throw new Error("Runner ownership or placement changed.");
  const storage=object(p.storageProfile),os=object(storage.osDisk),disk=object(os.managedDisk),nics=object(p.networkProfile).networkInterfaces,identity=object(v.identity);
  if(typeof disk.id!=="string" || !Array.isArray(nics) || nics.length!==1 || identity.type!=="SystemAssigned" || typeof identity.principalId!=="string" || os.diffDiskSettings || Array.isArray(storage.dataDisks)&&storage.dataDisks.length)throw new Error("Review persistent disk, single NIC and system identity before resize.");
  const preserved=digest({disk:disk.id,nic:object(nics[0]).id,principal:identity.principalId,tenant:identity.tenantId,controller:storage.diskControllerType,security:p.securityProfile,location:v.location,zones:v.zones});
  const statuses=object(p.instanceView).statuses;
  const power=Array.isArray(statuses)?statuses.map(x=>object(x).code).find(x=>typeof x==="string"&&x.startsWith("PowerState/")):undefined;
  return {preserved,power,size:String(object(p.hardwareProfile).vmSize),provisioning:p.provisioningState};
}
function gate(r:RunnerRecord){
  if(r.target?.phase!=="provisioned" || r.migration || r.upgrade && r.upgrade.phase!=="finished" || r.guestCommand && ["submitted","unknown"].includes(r.guestCommand.phase))throw new Error("Reconcile target and guest operations before resizing the idle runner.");
  targetBudget(r.target.input);
}
async function submit(control:RunnerControl,r:RunnerRecord,path:string,method:"POST"|"PATCH",body?:unknown){
  await control.persist(r);
  try {const response=await control.request(r.input.subscriptionId,path,method,body);if(response.status<200||response.status>=300)throw new Error();}
  catch {r={...r,resize:{...r.resize!,unknown:true}};await control.persist(r);}
  return r;
}
/** Native approval and exclusive workflow lock are held by caller. Never changes
 * the source VM, NIC, identity or persistent disk, nor automatically replays. */
export async function startResize(control:RunnerControl,r:RunnerRecord):Promise<RunnerRecord>{
  gate(r);if(r.resize)throw new Error("Reconcile the retained resize instead of starting another.");assertIdleHealth(r);
  const observed=await inspect(control,r),size=r.target!.input.loaderSize;
  if(observed.power!=="PowerState/running" || observed.provisioning!=="Succeeded")throw new Error("Check the running idle VM before resize.");
  await preflightRunner(control,{...r.input,size});
  const next:RunnerRecord={...r,resize:{phase:observed.size===size?"finished":"deallocating",size,previousSize:observed.size,preservedSHA256:observed.preserved,startedAt:new Date().toISOString()}};
  if(observed.size===size){await control.persist(next);return next;}
  delete next.guestReady;
  return submit(control,next,`${r.vmId}/deallocate?api-version=2024-07-01`,"POST");
}
/** Pending phases are GET-only. Ready phases require a new explicit approval. */
export async function advanceResize(control:RunnerControl,r:RunnerRecord,approved=false):Promise<RunnerRecord>{
  gate(r);if(!r.resize)throw new Error("No retained resize.");
  const v=await inspect(control,r),p=r.resize;
  if(v.preserved!==p.preservedSHA256)throw new Error("VM disk, NIC, identity or security changed; no further mutation is allowed.");
  let phase=p.phase;
  if(phase==="deallocating" && v.power==="PowerState/deallocated")phase="ready-to-resize";
  else if(phase==="resizing" && v.size===p.size && v.provisioning==="Succeeded" && v.power==="PowerState/deallocated")phase="ready-to-start";
  else if(phase==="starting" && v.size===p.size && v.provisioning==="Succeeded" && v.power==="PowerState/running")phase="finished";
  if(phase!==p.phase){const next={...r,resize:{...p,phase,unknown:false}};await control.persist(next);return next;}
  if(approved && phase==="ready-to-resize"){
    if(v.power!=="PowerState/deallocated" || v.size!==p.previousSize)throw new Error("VM state changed before size update.");
    await preflightRunner(control,{...r.input,size:p.size});
    return submit(control,{...r,resize:{...p,phase:"resizing",unknown:false}},`${r.vmId}?api-version=2024-07-01`,"PATCH",{properties:{hardwareProfile:{vmSize:p.size}}});
  }
  if(approved && phase==="ready-to-start"){
    if(v.power!=="PowerState/deallocated" || v.size!==p.size)throw new Error("VM state changed before start.");
    return submit(control,{...r,resize:{...p,phase:"starting",unknown:false}},`${r.vmId}/start?api-version=2024-07-01`,"POST");
  }
  return r;
}
