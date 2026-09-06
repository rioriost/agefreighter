import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { sourceWorkflowDraft, RunnerRecord } from "../../core/runner";
import { developmentArtifact } from "../../core/runnerDevelopment";
import { assertUpgradeIdle, refreshUpgrade, submitUpgrade, upgradeScript } from "../../core/runnerUpgrade";
import { RunnerControl } from "../../core/runnerLifecycle";
import { dispatchGuest } from "../../core/runnerGuest";
const id="11111111-1111-4111-8111-111111111111",boot="22222222-2222-4222-8222-222222222222";
function fixture() {
  const record=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
  record.phase="provisioned";record.artifact={version:"2.4.0",url:"https://example.invalid",sha256:"a".repeat(64)};
  record.guestReady={bootId:boot,cliVersion:record.artifact.version,archiveSha256:record.artifact.sha256,commit:"old",checkedAt:new Date().toISOString()};
  const artifact=developmentArtifact(record,{schemaVersion:1,platform:"linux-amd64",version:`2.4.0-dev.${"b".repeat(12)}`,commit:"b".repeat(40),sha256:"c".repeat(64),bytes:64});
  const events:string[]=[],saved:RunnerRecord[]=[];let view:unknown, fail=false;
  const control:RunnerControl={list:async()=>[],sleep:async()=>{},persist:async r=>{events.push("persist");saved.push(structuredClone(r));},request:async(_s,_p,m="GET")=>{
    events.push(m);if(m==="PUT"){if(fail)throw new Error("secret");return {status:202,value:{}};}
    return view?{status:200,value:{properties:{instanceView:view}}}:{status:404,value:{}};
  }};
  return {record,artifact,control,events,saved,result:(v:unknown)=>{view=v;},fail:()=>{fail=true;}};
}
test("upgrade pins bytes, preserves data, records intent before PUT and requires fresh readiness afterward",async()=>{
  const f=fixture(),n=await submitUpgrade(f.control,f.record,f.artifact);
  assert.deepEqual(f.events,["GET","persist","PUT"]);assert.equal(n.guestReady,undefined);assert.deepEqual(n.artifact,f.record.artifact);
  const script=upgradeScript(f.record,n.upgrade!);
  assert.match(script,/previous-agefreighter/);assert.match(script,/noclobber/);assert.match(script,/upgrade.lock/);assert.match(script,/sha256sum --check/);
  assert.doesNotMatch(script,/rm -rf|sig=|AccountKey|git clone/);
  if(process.platform!=="win32")execFileSync("bash",["-n"],{input:script});
  await assert.rejects(dispatchGuest(f.control,n,{version:1,workflow:id,operation:boot,action:"ready"}),/upgrade/);
  const ready={version:1,ready:true,os:"linux",architecture:"amd64",bootId:boot,cliVersion:f.artifact.version,archiveSha256:f.artifact.sha256,commit:f.artifact.development!.commit};
  f.result({executionState:"Succeeded",exitCode:0,output:JSON.stringify(ready)});
  const r=await refreshUpgrade(f.control,n);assert.equal(r.upgrade?.phase,"finished");assert.deepEqual(r.artifact,f.artifact);assert.equal(r.guestReady,undefined);
});
test("unknown upgrade never replays; foreign or incomplete receipts never switch artifact",async()=>{
  for(const mutation of [{bootId:id},{commit:"wrong"},{archiveSha256:"a".repeat(64)},{ready:false}]){
    const f=fixture();f.fail();const n=await submitUpgrade(f.control,f.record,f.artifact);assert.equal(n.upgrade?.phase,"unknown");
    const unchanged=await refreshUpgrade(f.control,n);assert.deepEqual(unchanged,n);assert.equal(f.events.filter(e=>e==="PUT").length,1);
    await assert.rejects(submitUpgrade(f.control,n,f.artifact),/Reconcile/);
    f.result({executionState:"Succeeded",exitCode:0,output:JSON.stringify({version:1,ready:true,os:"linux",architecture:"amd64",bootId:boot,cliVersion:f.artifact.version,archiveSha256:f.artifact.sha256,commit:f.artifact.development!.commit,...mutation})});
    const r=await refreshUpgrade(f.control,n);assert.equal(r.upgrade?.phase,"failed");assert.deepEqual(r.artifact,f.record.artifact);
  }
});
test("stale readiness, active work, artifact collision, and command limit block before mutation",async()=>{
  const f=fixture();f.record.guestReady!.checkedAt="2020-01-01T00:00:00Z";assert.throws(()=>assertUpgradeIdle(f.record),/Fresh/);
  f.record.guestReady!.checkedAt=new Date().toISOString();f.record.guestCommand={id:"pending",operation:boot,action:"status",phase:"unknown",submittedAt:new Date().toISOString()};assert.throws(()=>assertUpgradeIdle(f.record),/Reconcile/);
  delete f.record.guestCommand;f.control.list=async()=>Array.from({length:25},()=>({}));await assert.rejects(submitUpgrade(f.control,f.record,f.artifact),/limit/);assert.equal(f.saved.length,0);
});
