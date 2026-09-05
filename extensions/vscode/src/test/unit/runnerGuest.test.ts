import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { assembleGuestReport, dispatchGuest, guestDispatchScript, reconcileGuest } from "../../core/runnerGuest";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";

const id="11111111-1111-4111-8111-111111111111", op="22222222-2222-4222-8222-222222222222";
function record(): RunnerRecord {
  return { schemaVersion:2,id,phase:"provisioned",input:{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_B2s_v2",source:{type:"neo4j",location:"on-premises"}},artifact:{version:"2.4.0",sha256:"a".repeat(64),url:"https://example.invalid/artifact"},vmId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/runner`,deploymentId:"deployment",template:{},previewHash:"hash",expiresAt:"",updatedAt:"",hourlyComputeUSD:.1 };
}
function fixture() {
  const events:string[]=[],saved:RunnerRecord[]=[], bodies:unknown[]=[];
  let fail=false, result:unknown=undefined;
  const control:RunnerControl={sleep:async()=>{},list:async()=>[],persist:async r=>{events.push("persist");saved.push(structuredClone(r));},request:async(_sub,path,method="GET",body)=>{
    events.push(method+":"+path);
    if(method==="PUT"){ bodies.push(body);if(fail)throw new Error("sensitive remote failure");return {status:201,value:{}}; }
    if(result!==undefined)return {status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify(result)}}}};
    return {status:404,value:{}};
  }};
  return {control,events,saved,bodies,fail:()=>{fail=true;},result:(r:unknown)=>{result=r;}};
}
test("protected dispatch records intent before PUT and never persists secrets",async()=>{
  const f=fixture(),r=record();r.guestReady={bootId:op,cliVersion:"2.4.0",archiveSha256:r.artifact.sha256,commit:"commit",checkedAt:new Date().toISOString()};
  const submitted=await dispatchGuest(f.control,r,{version:1,workflow:id,operation:op,action:"profile",configuration:{source:"reviewed"},secrets:{AGEFREIGHTER_SOURCE_PASSWORD:"never-public"}});
  assert.equal(submitted.guestCommand?.phase,"submitted");assert.equal(f.events[1],"persist");assert.ok(f.events[2]!.startsWith("PUT:"));
  assert.ok(!JSON.stringify(f.saved).includes("never-public"));assert.ok(!guestDispatchScript.includes("never-public"));
  const body=f.bodies[0] as {properties:{protectedParameters:{value:string}[];parameters?:unknown}};
  assert.equal(body.properties.parameters,undefined);assert.match(Buffer.from(body.properties.protectedParameters[0]!.value,"base64").toString(),/never-public/);
  await assert.rejects(dispatchGuest(f.control,submitted,{version:1,workflow:id,operation:op,action:"profile",configuration:{}}),/Reconcile/);
});
test("ambiguous transport results reconcile with GET only",async()=>{
  const f=fixture();f.fail();const next=await dispatchGuest(f.control,record(),{version:1,workflow:id,operation:op,action:"ready"});
  assert.equal(next.guestCommand?.phase,"unknown");const before=f.events.length;await reconcileGuest(f.control,next);assert.ok(f.events.slice(before).every(e=>e.startsWith("GET:")));
});
test("readiness requires matching Linux architecture, release and checksum",async()=>{
  const f=fixture();const next=await dispatchGuest(f.control,record(),{version:1,workflow:id,operation:op,action:"ready"});
  const ready={version:1,ready:true,os:"linux",architecture:"amd64",bootId:op,cliVersion:"2.4.0",archiveSha256:next.artifact.sha256,commit:"commit"};
  f.result(ready);assert.equal((await reconcileGuest(f.control,next)).record.guestReady?.bootId,op);
  for(const change of [{architecture:"arm64"},{os:"darwin"},{archiveSha256:"wrong"},{cliVersion:"2.3.0"},{ready:false}]){
    f.result({...ready,...change});const checked=await reconcileGuest(f.control,next);assert.equal(checked.record.guestCommand?.phase,"failed");assert.equal(checked.record.guestReady,undefined);
  }
});
test("all report chunks must agree on operation, length, offset and digest",()=>{
  const data=Buffer.from(JSON.stringify({sample:"日本語".repeat(500)})),sha=createHash("sha256").update(data).digest("hex");
  const chunks:unknown[]=[];for(let offset=0;offset<data.length;offset+=1536)chunks.push({version:1,operation:op,offset,total:data.length,sha256:sha,data:data.subarray(offset,offset+1536).toString("base64")});
  assert.equal(assembleGuestReport(op,chunks,sha),data.toString());
  assert.throws(()=>assembleGuestReport(op,chunks.slice(1),sha));assert.throws(()=>assembleGuestReport(op,[...chunks].reverse(),sha));assert.throws(()=>assembleGuestReport(op,chunks,"b".repeat(64)));assert.throws(()=>assembleGuestReport(id,chunks,sha));
});
test("assessment requires recent matching readiness and pins its boot identity",async()=>{
  const f=fixture(),r=record();
  const ready={bootId:op,cliVersion:"2.4.0",archiveSha256:r.artifact.sha256,commit:"commit",checkedAt:new Date().toISOString()};
  const request={version:1 as const,workflow:id,operation:op,action:"profile" as const,configuration:{},expectedBootId:"untrusted"};
  for(const change of [{checkedAt:new Date(Date.now()-301000).toISOString()},{checkedAt:"invalid"},{checkedAt:new Date(Date.now()+60000).toISOString()},{archiveSha256:"wrong"},{bootId:"invalid"}]){
    r.guestReady={...ready,...change};await assert.rejects(dispatchGuest(f.control,r,request),/fresh guest readiness/);
  }
  assert.equal(f.events.length,0);
  r.guestReady=ready;await dispatchGuest(f.control,r,request);
  const body=f.bodies[0] as {properties:{protectedParameters:{value:string}[]}};
  assert.equal(JSON.parse(Buffer.from(body.properties.protectedParameters[0]!.value,"base64").toString()).expectedBootId,op);
});
test("re-reading readiness cannot renew an old check",async()=>{
  const f=fixture();const next=await dispatchGuest(f.control,record(),{version:1,workflow:id,operation:op,action:"ready"});
  next.guestCommand!.submittedAt="2020-01-01T00:00:00Z";
  f.result({version:1,ready:true,os:"linux",architecture:"amd64",bootId:op,cliVersion:"2.4.0",archiveSha256:next.artifact.sha256,commit:"commit"});
  const checked=(await reconcileGuest(f.control,next)).record;
  assert.equal(checked.guestReady?.checkedAt,"2020-01-01T00:00:00Z");
  await assert.rejects(dispatchGuest(f.control,checked,{version:1,workflow:id,operation:op,action:"inventory",configuration:{}}),/fresh guest readiness/);
});
