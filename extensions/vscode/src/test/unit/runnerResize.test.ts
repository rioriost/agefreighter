import assert from "node:assert/strict";
import test from "node:test";
import {sourceWorkflowDraft,RunnerRecord} from "../../core/runner";
import {RunnerControl} from "../../core/runnerLifecycle";
import {startResize,advanceResize} from "../../core/runnerResize";
import {applyTargetPreload} from "../../core/runnerExecution";

function fixture(){
  const id="11111111-1111-4111-8111-111111111111",base=`/subscriptions/${id}/resourceGroups/test/providers`,vnet=`${base}/Microsoft.Network/virtualNetworks/net`;
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:vnet+"/subnets/runner",size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
  r.phase="provisioned";r.artifact.version="dev";r.artifact.sha256="a".repeat(64);
  r.guestReady={bootId:id,cliVersion:"dev",archiveSha256:r.artifact.sha256,commit:"b".repeat(40),checkedAt:new Date().toISOString(),health:{idle:true,storageUsedPercent:6,swapUsedBytes:0,oomEvents:0}};
  r.target={phase:"provisioned",serverId:base+"/Microsoft.DBforPostgreSQL/flexibleServers/test",input:{loaderSize:"Standard_D4s_v5",hourlyUSD:1,additionalReserveUSD:100,budgetUSD:800,deadline:new Date(Date.now()+86400000).toISOString()}} as any;
  const vm={location:"japaneast",zones:["1"],identity:{type:"SystemAssigned",principalId:id,tenantId:id},tags:{application:"agefreighter",workflow:id,purpose:"discovery-and-migration"},properties:{provisioningState:"Succeeded",hardwareProfile:{vmSize:"Standard_B2s_v2"},instanceView:{statuses:[{code:"PowerState/running"}]},networkProfile:{networkInterfaces:[{id:base+"/Microsoft.Network/networkInterfaces/runner"}]},storageProfile:{osDisk:{managedDisk:{id:base+"/Microsoft.Compute/disks/runner"}},diskControllerType:"SCSI",dataDisks:[]}}};
  const events:string[]=[],saved:RunnerRecord[]=[];
  const control:RunnerControl={persist:async x=>{events.push("persist");saved.push(structuredClone(x));},sleep:async()=>{},request:async(_s,path,method="GET")=>{
    events.push(method);
    if(method!=="GET")throw new Error("lost acknowledgement");
    if(path.startsWith(r.vmId+"?"))return {status:200,value:vm};
    if(path.startsWith(vnet+"?"))return {status:200,value:{location:"japaneast"}};
    return {status:200,value:{properties:{delegations:[]}}};
  },list:async(_s,path)=>path.includes("/skus?")?[{name:"Standard_D4s_v5",resourceType:"virtualMachines",family:"standardDSv5Family",locations:["japaneast"],locationInfo:[{location:"japaneast",zones:["1"]}],capabilities:[{name:"vCPUs",value:"4"},{name:"MemoryGB",value:"16"}],restrictions:[]}]:["cores","standardDSv5Family"].map(value=>({name:{value},currentValue:0,limit:64}))};
  return {r,vm,control,events,saved};
}
test("same-VM resize seals identity before mutation and unknown steps reconcile without replay",async()=>{
  const f=fixture();let r=await startResize(f.control,f.r);
  assert.equal(r.resize?.phase,"deallocating");assert.equal(r.resize?.unknown,true);assert.equal(r.guestReady,undefined);
  assert.ok(f.events.indexOf("persist")<f.events.indexOf("POST"));
  const posts=()=>f.events.filter(x=>x==="POST"||x==="PATCH").length;
  await advanceResize(f.control,r,true);assert.equal(posts(),1);
  f.vm.properties.instanceView.statuses[0]!.code="PowerState/deallocated";
  r=await advanceResize(f.control,r,true);assert.equal(r.resize?.phase,"ready-to-resize");assert.equal(posts(),1);
  r=await advanceResize(f.control,r,true);assert.equal(r.resize?.phase,"resizing");assert.equal(posts(),2);
  await advanceResize(f.control,r,true);assert.equal(posts(),2);
  f.vm.properties.hardwareProfile.vmSize="Standard_D4s_v5";r=await advanceResize(f.control,r);assert.equal(r.resize?.phase,"ready-to-start");
  r=await advanceResize(f.control,r,true);assert.equal(posts(),3);
  f.vm.properties.instanceView.statuses[0]!.code="PowerState/running";r=await advanceResize(f.control,r);assert.equal(r.resize?.phase,"finished");
  assert.equal(r.input.size,"Standard_B2s_v2");assert.equal(r.resize?.size,"Standard_D4s_v5");
});
test("resize refuses missing health, expired budget and changed persistent identity",async()=>{
  const a=fixture();delete a.r.guestReady!.health;await assert.rejects(startResize(a.control,a.r),/readiness/);assert.ok(!a.events.includes("POST"));
  const b=fixture();b.r.target!.input.deadline="2000-01-01T00:00:00Z";await assert.rejects(startResize(b.control,b.r),/budget|deadline/);
  const c=fixture();const r=await startResize(c.control,c.r);c.vm.identity.principalId="foreign";await assert.rejects(advanceResize(c.control,r,true),/changed/);
});
test("AGE preload restart is once-only and requires owned configuration to reconcile",async()=>{
  const f=fixture();let pending=true;const events:string[]=[];
  f.control.request=async(_s,path,method="GET")=>{
    events.push(method);if(method==="POST")throw Error("unknown");
    return {status:200,value:path.includes("/configurations/")?{properties:{value:"pg_stat_statements,age",isConfigPendingRestart:pending}}:{tags:{application:"agefreighter",purpose:"csv-migration-target",workflow:f.r.id},properties:{state:"Ready"}}};
  };
  let r=await applyTargetPreload(f.control,f.r,true);assert.equal(r.targetRestart?.phase,"unknown");
  r=await applyTargetPreload(f.control,r,true);assert.equal(events.filter(x=>x==="POST").length,1);
  pending=false;r=await applyTargetPreload(f.control,r);assert.equal(r.targetRestart?.phase,"finished");
});
