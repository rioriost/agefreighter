import assert from "node:assert/strict";
import test from "node:test";
import { sourceWorkflowDraft, RunnerRecord } from "../../core/runner";
import { reportStorageNames } from "../../core/runnerReportStorage";
import { storageDraft, submitStorage, refreshStorage } from "../../core/runnerStorageLifecycle";
import { RunnerControl } from "../../core/runnerLifecycle";
import { issueReportCapability } from "../../guided/blobCapabilities";
import { reportCapability } from "../../core/runnerBlob";

const id="11111111-1111-4111-8111-111111111111", op="22222222-2222-4222-8222-222222222222";
function fixture() {
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
  r.storageDeployment=storageDraft(r,op);
  const events:string[]=[], saved:RunnerRecord[]=[];let fail=false,existing=false,state="Succeeded",extra=false;
  const control:RunnerControl={list:async()=>[],sleep:async()=>{},persist:async r=>{events.push("persist");saved.push(structuredClone(r));},request:async(_sub,_path,method="GET")=>{
    events.push(method);
    if(method==="PUT"){if(fail)throw new Error("lost response");return{status:201,value:{}};}
    if(method==="POST"){const names=reportStorageNames(r);return{status:200,value:{status:"Succeeded",properties:{changes:[names.id,names.containerId,r.storageDeployment!.roleId,...extra?["/foreign"]:[]].map(resourceId=>({resourceId,changeType:"Create"}))}}};}
    return existing?{status:200,value:{properties:{provisioningState:state}}}:{status:404,value:{}};
  }};
  return{r,control,events,saved,fail:()=>{fail=true;},exists:()=>{existing=true;},extra:()=>{extra=true;},state:(s:string)=>{state=s;}};
}
test("storage preview creates only owned non-anonymous LRS storage and one account-scoped user role",()=>{
  const f=fixture(),d=f.r.storageDeployment!,resources=d.template.resources as any[],names=reportStorageNames(f.r);
  assert.equal(resources.length,3);assert.equal(resources[0].properties.allowSharedKeyAccess,false);assert.equal(resources[0].properties.allowBlobPublicAccess,false);
  assert.equal(resources[2].scope,names.id);assert.equal(resources[2].properties.principalId,op);assert.equal(resources[2].properties.principalType,"User");
  assert.doesNotMatch(JSON.stringify(d),/accountKey|sig=|password/);
});
test("storage PUT is persist-first and uncertain submission can only reconcile",async()=>{
  const f=fixture();f.fail();const r=await submitStorage(f.control,f.r);
  assert.equal(r.storageDeployment?.phase,"unknown");assert.equal(f.events.at(-3),"persist");assert.equal(f.events.at(-2),"PUT");
  await assert.rejects(submitStorage(f.control,r));f.exists();const before=f.events.length;
  const ready=await refreshStorage(f.control,r);assert.equal(ready.storageDeployment?.phase,"ready");assert.deepEqual(f.events.slice(before),["GET","persist"]);
});
test("storage refuses existing resources, foreign what-if changes and stale or modified approvals",async()=>{
  for(const kind of ["existing","extra","expired","tampered"]){const f=fixture();if(kind==="existing")f.exists();if(kind==="extra")f.extra();if(kind==="expired")f.r.storageDeployment!.expiresAt="2000-01-01T00:00:00Z";if(kind==="tampered")f.r.storageDeployment!.principalId=id;
    await assert.rejects(submitStorage(f.control,f.r));assert.ok(!f.events.includes("PUT"));}
});
test("actual Azure SDK issues a scoped user-delegation SAS through one bounded no-redirect request",async()=>{
  const f=fixture(),names=reportStorageNames(f.r);let calls=0;
  const now=Date.now(),date=(offset:number)=>new Date(now+offset).toISOString();
  const xml=`<?xml version="1.0" encoding="utf-8"?><UserDelegationKey><SignedOid>${op}</SignedOid><SignedTid>${id}</SignedTid><SignedStart>${date(-120000)}</SignedStart><SignedExpiry>${date(720000)}</SignedExpiry><SignedService>b</SignedService><SignedVersion>2023-11-03</SignedVersion><Value>${Buffer.alloc(32,1).toString("base64")}</Value></UserDelegationKey>`;
  const url=await issueReportCapability(f.r,op,"c",{getToken:async scope=>{assert.ok(String(scope).includes("https://storage.azure.com/.default"));return{token:"private-storage-token",expiresOnTimestamp:now+3600000};}},async (url,init)=>{
    calls++;assert.equal(String(url),`${names.origin}/?restype=service&comp=userdelegationkey`);assert.equal(init?.redirect,"error");assert.equal(init?.method,"POST");assert.ok(init?.signal);
    return new Response(xml,{headers:{"content-type":"application/xml"}});
  });
  assert.equal(calls,1);const parsed=reportCapability(url,id,op,"c");assert.equal(parsed.searchParams.get("skoid"),op);assert.ok(parsed.searchParams.get("sig"));
  assert.ok(!url.includes("private-storage-token"));
});
test("delegation key HTTP errors never retry, fall back to keys, or reveal service diagnostics",async()=>{
  const f=fixture();let calls=0;
  await assert.rejects(issueReportCapability(f.r,op,"r",{getToken:async()=>({token:"secret",expiresOnTimestamp:Date.now()+3600000})},async()=>{calls++;return new Response("SECRET SERVICE RESPONSE",{status:403});}),e=>e instanceof Error&&!e.message.includes("SECRET SERVICE RESPONSE"));
  assert.equal(calls,1);
});
