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
test("target price selection rejects ambiguity, future rates, missing services and non-finite values",()=>{
  const {input}=fixture();const rates=[{armSkuName:input.loaderSize,serviceName:"Virtual Machines",hourlyUSD:.248,effectiveStartDate:"2023-01-01"},{armSkuName:input.postgresSKU,serviceName:"Azure Database for PostgreSQL",hourlyUSD:.488,effectiveStartDate:"2023-01-01"}];
  assert.equal(targetComputeRate(rates,input),.736);
  for(const values of [[rates[0]!],[...rates,rates[1]!],[rates[0]!,{...rates[1]!,hourlyUSD:NaN}],[rates[0]!,{...rates[1]!,effectiveStartDate:"2999-01-01"}]])assert.throws(()=>targetComputeRate(values,input),/unique/);
});
