import assert from "node:assert/strict";
import test from "node:test";
import { credentialBinding, credentialKey, rememberSourceCredential, savedSourceCredential } from "../../core/sourceCredential";
import { boundedWatch } from "../../core/boundedWatch";
import { retainTargetDraft, targetDraftBinding } from "../../core/targetDraft";
import { catalogFixture, catalogForm } from "../catalogFixtures";

function vault() {
  const data=new Map<string,string>();
  return {data,get:async(k:string)=>data.get(k),store:async(k:string,v:string)=>{data.set(k,v);},delete:async(k:string)=>{data.delete(k);}};
}
test("source password is opt-in encrypted-store only and survives reloaded record with same connection",async()=>{
  const {record:r}=catalogFixture(),v=vault(),now=1000000;
  assert.equal(await savedSourceCredential(v,r,catalogForm,now),undefined);
  await rememberSourceCredential(v,r,catalogForm,"PRIVATE",now);
  assert.equal(await savedSourceCredential(v,structuredClone(r),catalogForm,now+1),"PRIVATE");
  assert.ok(!JSON.stringify(r).includes("PRIVATE"));assert.equal(v.data.size,1);
});
for(const field of ["host","port","database","username","ca","workflow","placement","type","expiry"]){test(`saved credential refuses changed ${field}`,async()=>{
  const {record:r}=catalogFixture(),v=vault(),c={...catalogForm},now=1000000;
  await rememberSourceCredential(v,r,c,"PRIVATE",now);
  if(field==="host")c.host="another.example.com";
  if(field==="port")c.port++;
  if(field==="database")c.database="other";
  if(field==="username")c.username="admin";
  if(field==="ca")r.sourceCA={path:"/ca",name:"ca",bytes:1,sha256:"a".repeat(64)};
  if(field==="workflow")r.id="22222222-2222-4222-8222-222222222222";
  if(field==="placement")r.input.source.location="other-cloud";
  if(field==="type")r.input.source.type="neo4j";
  assert.equal(await savedSourceCredential(v,r,c,now+(field==="expiry"?8*3600000:1)),undefined);
});}
test("secret expiry does not extend target authorization; invalid vault content is removed",async()=>{
  const {record:r}=catalogFixture(),v=vault(),now=1000000;
  r.target={input:{deadline:new Date(now+1000).toISOString()}} as any;
  await rememberSourceCredential(v,r,catalogForm,"PRIVATE",now);
  assert.equal(await savedSourceCredential(v,r,catalogForm,now+1000),undefined);
  await assert.rejects(()=>rememberSourceCredential(v,r,catalogForm,"PRIVATE",now+1000),/expired/);
  v.data.set(credentialKey(r.id),"malformed PRIVATE");
  assert.equal(await savedSourceCredential(v,r,catalogForm,now),undefined);assert.equal(v.data.size,0);
  assert.throws(()=>credentialBinding(r,{...catalogForm,port:0}),/connection/);
});
test("retained failure invalidates saved credential even if the observing panel was closed",async()=>{
  const {record:r}=catalogFixture(),v=vault();
  await rememberSourceCredential(v,r,catalogForm,"PRIVATE");
  r.assessment={operation:"failed",phase:"failed",action:"inventory",bootId:"boot",configurationSHA256:"a".repeat(64)};
  assert.equal(await savedSourceCredential(v,r,catalogForm),undefined);assert.equal(v.data.size,0);
});
test("archiving a failed attempt cannot resurrect its previously cached password",async()=>{
  const {record:r}=catalogFixture(),v=vault();
  await rememberSourceCredential(v,r,catalogForm,"PRIVATE");
  r.assessmentHistory=[{operation:"failed",phase:"failed",action:"inventory",bootId:"boot",configurationSHA256:"a".repeat(64)}];
  assert.equal(await savedSourceCredential(v,r,catalogForm),undefined);assert.equal(v.data.size,0);
});
test("target draft saves without live readiness or Azure requests and refuses changed binding",()=>{
  const {record:r}=catalogFixture();delete r.guestReady;
  const binding=targetDraftBinding(r),next=retainTargetDraft(r,binding,{subnetCIDR:"10.0.24.0/24"},"/chosen");
  assert.equal(next.target,undefined);assert.equal(next.targetDraft?.folder,"/chosen");
  assert.equal(next.targetDraft?.input.subnetCIDR,"10.0.24.0/24");
  assert.equal(targetDraftBinding(next),binding);
  next.input.region="westus";assert.throws(()=>retainTargetDraft(next,binding,{}),/changed/);
});
test("watch stops at terminal and holds no work after completion",async()=>{
  let n=0,time=0,sleeps=0;
  const result=await boundedWatch({step:async()=>++n,done:n=>n===3,sleep:async ms=>{sleeps++;time+=ms;},now:()=>time,deadline:9000,intervalMs:1000,maxSteps:10});
  assert.equal(result,3);assert.equal(n,3);assert.equal(sleeps,2);
});
test("watch errors are never retried",async()=>{
  let n=0;
  await assert.rejects(()=>boundedWatch({step:async()=>{n++;throw Error("uncertain");},done:()=>false,sleep:async()=>{},deadline:Date.now()+10000,intervalMs:1000,maxSteps:10}),/uncertain/);
  assert.equal(n,1);
});
test("watch respects cancellation, hard deadline and command limit",async()=>{
  for(const mode of ["cancel","deadline","limit"]){
    let n=0,time=0;
    await boundedWatch({step:async()=>++n,done:()=>false,sleep:async ms=>{time+=ms;},now:()=>time,cancelled:()=>mode==="cancel",deadline:mode==="deadline"?1500:100000,intervalMs:1000,maxSteps:3});
    assert.equal(n,mode==="cancel"?0:mode==="deadline"?2:3);
  }
});
