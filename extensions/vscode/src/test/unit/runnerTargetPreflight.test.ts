import assert from "node:assert/strict";
import test from "node:test";
import { preflightTarget, targetComputeRate } from "../../core/runnerTargetPreflight";
import { sourceWorkflowDraft } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { TargetInput } from "../../core/runnerTarget";

function fixture(){
  const id="11111111-1111-4111-8111-111111111111",base=`/subscriptions/${id}/resourceGroups/test/providers`,subnet=`${base}/Microsoft.Network/virtualNetworks/net/subnets/runner`,nic=`${base}/Microsoft.Network/networkInterfaces/runner`;
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:subnet,size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
  r.phase="provisioned";r.artifact.version="dev";r.artifact.sha256="a".repeat(64);r.guestReady={bootId:id,cliVersion:"dev",archiveSha256:r.artifact.sha256,commit:"b".repeat(40),checkedAt:new Date().toISOString()};
  const input:TargetInput={serverName:"csv-test",subnetCIDR:"10.0.2.0/24",postgresSKU:"Standard_D4ds_v5",postgresTier:"GeneralPurpose",storageGiB:128,loaderSize:"Standard_D4s_v5",hourlyUSD:1,additionalReserveUSD:100,budgetUSD:800,deadline:new Date(Date.now()+86400000).toISOString()};
  const vm={location:"japaneast",zones:["1"],tags:{application:"agefreighter",workflow:id,purpose:"discovery-and-migration"},properties:{provisioningState:"Succeeded",instanceView:{statuses:[{code:"PowerState/running"}]},networkProfile:{networkInterfaces:[{id:nic}]}}};
  const quota=(names:string[])=>names.map(value=>({name:{value},currentValue:0,limit:64}));
  const calls:string[]=[];
  const control:RunnerControl={persist:async()=>{throw Error("read only");},sleep:async()=>{},request:async(_s,path,method="GET")=>{
    assert.equal(method,"GET");calls.push(path);
    const value=path.startsWith(r.vmId+"?")?vm:path.startsWith(nic+"?")?{properties:{ipConfigurations:[{properties:{subnet:{id:subnet}}}]}}:path.startsWith(subnet+"?")?{properties:{delegations:[]}}:path.includes("virtualNetworks/net?")?{location:"japaneast",properties:{addressSpace:{addressPrefixes:["10.0.0.0/16"]},subnets:[{properties:{addressPrefix:"10.0.1.0/24"}}]}}:{};
    return {status:200,value};
  },list:async(_s,path)=>{
    calls.push(path);
    if(path.includes("/capabilities?"))return [{supportedServerVersions:[{name:"18"}],supportedServerEditions:[{name:"GeneralPurpose",supportedStorageEditions:[{supportedStorageMb:[{storageSizeMb:131072}]}],supportedServerSkus:[{name:"Standard_D4ds_v5",vCores:4,supportedMemoryPerVcoreMb:4096,supportedZones:["1"]}]}]}];
    if(path.includes("/skus?"))return [{name:"Standard_D4s_v5",resourceType:"virtualMachines",family:"standardDSv5Family",locations:["japaneast"],locationInfo:[{location:"japaneast",zones:["1"]}],capabilities:[{name:"vCPUs",value:"4"},{name:"MemoryGB",value:"16"}],restrictions:[]}];
    return quota(path.includes("Microsoft.Compute")?["cores","standardDSv5Family"]:["cores","standardDDSv5Family"]);
  }};
  return {r,input,vm,control,calls};
}
function settlingFixture(){
  const f=fixture();
  f.r.guestCommand={id:`${f.r.vmId}/runCommands/af-${f.r.id}`,operation:f.r.id,action:"ready",phase:"finished",submittedAt:f.r.guestReady!.checkedAt};
  f.vm.properties.provisioningState="Updating";
  return f;
}
test("completed readiness waits for parent VM ARM completion using only GETs",async()=>{
  const f=settlingFixture(),before=JSON.stringify(f.r),progress:string[]=[];
  let sleeps=0;
  f.control.sleep=async ms=>{assert.equal(ms,3000);if(++sleeps===2)f.vm.properties.provisioningState="Succeeded";};
  await preflightTarget(f.control,f.r,f.input,{progress:message=>progress.push(message)});
  assert.equal(sleeps,2);assert.equal(progress.length,2);assert.match(progress[0]!,/status reads only/);
  assert.equal(f.calls.filter(p=>p.startsWith(f.r.vmId+"?")).length,3);
  assert.equal(f.calls.filter(p=>p.includes("/capabilities?")).length,1);
  assert.equal(JSON.stringify(f.r),before);
});
test("persistent Updating is bounded and preserves the saved workflow without deployment or readiness replay",async()=>{
  const f=settlingFixture(),before=JSON.stringify(f.r);let sleeps=0;
  f.control.sleep=async()=>{sleeps++;};
  await assert.rejects(preflightTarget(f.control,f.r,f.input),/still Updating.*No target deployment.*saved target inputs/);
  assert.equal(sleeps,20);assert.equal(f.calls.filter(p=>p.startsWith(f.r.vmId+"?")).length,21);
  assert.ok(!f.calls.some(p=>p.includes("/capabilities?")||p.includes("/runCommands")));
  assert.equal(JSON.stringify(f.r),before);
});
test("VM polling stops at the elapsed-time bound even if one GET is slow",async t=>{
  const f=settlingFixture(),request=f.control.request;
  t.mock.timers.enable({apis:["Date"],now:Date.now()});
  let reads=0;
  f.control.request=async(...args)=>{
    if(args[1].startsWith(f.r.vmId+"?")){reads++;t.mock.timers.tick(60000);}
    return request(...args);
  };
  f.control.sleep=async()=>assert.fail("No further wait after the deadline");
  await assert.rejects(preflightTarget(f.control,f.r,f.input),/still Updating/);
  assert.equal(reads,1);
});
for(const state of ["Failed","Deleting","Creating","unknown"])test(`target preflight does not poll terminal or unrelated VM state ${state}`,async()=>{
  const f=settlingFixture();f.vm.properties.provisioningState=state;
  f.control.sleep=async()=>assert.fail("No retry of unrelated state");
  await assert.rejects(preflightTarget(f.control,f.r,f.input),/Runner provisioning has not succeeded/);
});
for(const change of ["missing","pending","other action","foreign VM","different receipt"] as const)test(`Updating requires exact completed readiness: ${change}`,async()=>{
  const f=settlingFixture();
  if(change==="missing")delete f.r.guestCommand;
  if(change==="pending")f.r.guestCommand!.phase="submitted";
  if(change==="other action")f.r.guestCommand!.action="status";
  if(change==="foreign VM")f.r.guestCommand!.id="/other-vm/runCommands/af-"+f.r.id;
  if(change==="different receipt")f.r.guestCommand!.submittedAt="2020-01-01T00:00:00Z";
  f.control.sleep=async()=>assert.fail("Unrelated updates must not be polled");
  await assert.rejects(preflightTarget(f.control,f.r,f.input),/Runner provisioning has not succeeded/);
});
for(const change of ["ownership","region","zone","power"] as const)test(`VM ${change} is revalidated on every polling read`,async()=>{
  const f=settlingFixture();let sleeps=0;
  f.control.sleep=async()=>{
    sleeps++;
    if(change==="ownership")f.vm.tags.workflow="foreign";
    if(change==="region")f.vm.location="japanwest";
    if(change==="zone")f.vm.zones=["2"];
    if(change==="power")f.vm.properties.instanceView.statuses=[{code:"PowerState/deallocated"}];
  };
  await assert.rejects(preflightTarget(f.control,f.r,f.input),change==="power"?/running guest/:/ownership or placement/);
  assert.equal(sleeps,1);
});
for(const boundary of ["before read","during read","during wait"] as const)test(`target wait cancellation ${boundary} stops further reads`,async()=>{
  const f=settlingFixture(),request=f.control.request;let cancelled=boundary==="before read",reads=0;
  f.control.request=async(...args)=>{
    if(args[1].startsWith(f.r.vmId+"?")){reads++;if(boundary==="during read")cancelled=true;}
    return request(...args);
  };
  f.control.sleep=async()=>{cancelled=true;};
  await assert.rejects(preflightTarget(f.control,f.r,f.input,{cancelled:()=>cancelled}),/Target review cancelled/);
  assert.equal(reads,boundary==="before read"?0:1);
});
for(const gate of ["readiness","budget"] as const)test(`target wait cannot outlive ${gate} authorization`,async()=>{
  const f=settlingFixture();let sleeps=0;
  f.control.sleep=async()=>{
    sleeps++;
    if(gate==="readiness")f.r.guestReady!.checkedAt="2020-01-01T00:00:00Z";
    else f.input.deadline="2020-01-01T00:00:00Z";
  };
  await assert.rejects(preflightTarget(f.control,f.r,f.input),gate==="readiness"?/Fresh matching/:/budget or deadline/);
  assert.equal(sleeps,1);assert.equal(f.calls.filter(p=>p.startsWith(f.r.vmId+"?")).length,1);
});
test("target preflight rechecks cancellation and readiness after capacity reads",async()=>{
  for(const gate of ["cancel","readiness"]){
    const f=fixture(),list=f.control.list;let cancelled=false;
    f.control.list=async(...args)=>{
      const result=await list(...args);
      if(args[1].includes("resourceType")){
        if(gate==="cancel")cancelled=true;else f.r.guestReady!.checkedAt="2020-01-01T00:00:00Z";
      }
      return result;
    };
    await assert.rejects(preflightTarget(f.control,f.r,f.input,{cancelled:()=>cancelled}),gate==="cancel"?/cancelled/:/Fresh matching/);
  }
});
test("target preflight reads ownership, private network, service capacity and both quotas without mutation",async()=>{
  const {r,input,control,calls}=fixture();await preflightTarget(control,r,input);
  assert.ok(calls.some(x=>x.includes("resourceType/flexibleServers/usages?api-version=2023-06-01-preview")));
});
test("target preflight rejects stale guest, changed ownership, network overlap and missing quotas",async()=>{
  const a=fixture();a.r.guestReady!.checkedAt="2020-01-01T00:00:00Z";await assert.rejects(preflightTarget(a.control,a.r,a.input),/Fresh/);
  const b=fixture();b.vm.tags.workflow="foreign";await assert.rejects(preflightTarget(b.control,b.r,b.input),/ownership/);
  const c=fixture();c.input.subnetCIDR="10.0.1.0/24";await assert.rejects(preflightTarget(c.control,c.r,c.input),/overlap/);
  const d=fixture(),list=d.control.list;d.control.list=async(s,p)=>p.includes("resourceType")?[]:list(s,p);await assert.rejects(preflightTarget(d.control,d.r,d.input),/quota/);
});
test("independent network group preflight reads both existing groups and keeps the NIC in the migration group",async()=>{
  const f=fixture(),request=f.control.request;
  f.r.input.subnetId=f.r.input.subnetId.replace("/resourceGroups/test/","/resourceGroups/network-only/");
  f.control.request=async(s,p,m,b)=>{
    if(p.includes("/networkInterfaces/"))return {status:200,value:{properties:{ipConfigurations:[{properties:{subnet:{id:f.r.input.subnetId}}}]}}};
    if(p.startsWith(f.r.input.subnetId+"?"))return {status:200,value:{properties:{delegations:[]}}};
    return request(s,p,m,b);
  };
  await preflightTarget(f.control,f.r,f.input);
  for(const group of ["test","network-only"])assert.ok(f.calls.some(x=>x.includes(`/resourceGroups/${group}?`)));
  for(const status of [403,404,500]){
    const read=f.control.request;
    const control={...f.control,request:async(s:string,p:string,m?:"GET"|"POST"|"PUT"|"PATCH",b?:unknown)=>p.includes("/resourceGroups/network-only?")?{status,value:{}}:read(s,p,m,b)};
    await assert.rejects(preflightTarget(control,f.r,f.input),/Both migration and network/);
  }
});
test("cross-subscription network fails before any preflight calls",async()=>{
  const f=fixture();f.r.input.subnetId=f.r.input.subnetId.replace(f.r.input.subscriptionId,"22222222-2222-4222-8222-222222222222");
  await assert.rejects(preflightTarget(f.control,f.r,f.input),/runner subscription/);assert.equal(f.calls.length,0);
});
test("target price selection rejects ambiguity, future rates, missing services and non-finite values",()=>{
  const {input}=fixture();const rates=[{armSkuName:input.loaderSize,serviceName:"Virtual Machines",hourlyUSD:.248,effectiveStartDate:"2023-01-01"},{armSkuName:input.postgresSKU,serviceName:"Azure Database for PostgreSQL",hourlyUSD:.488,effectiveStartDate:"2023-01-01"}];
  assert.equal(targetComputeRate(rates,input),.736);
  for(const values of [[rates[0]!],[...rates,rates[1]!],[rates[0]!,{...rates[1]!,hourlyUSD:NaN}],[rates[0]!,{...rates[1]!,effectiveStartDate:"2999-01-01"}]])assert.throws(()=>targetComputeRate(values,input),/unique/);
});
