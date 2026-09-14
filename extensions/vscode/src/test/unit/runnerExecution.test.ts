import assert from "node:assert/strict";
import test from "node:test";
import {sourceWorkflowDraft} from "../../core/runner";
import {RunnerControl} from "../../core/runnerLifecycle";
import {migrationPreflight,refreshMigration,targetDSN,sameAzureLocation,waitForMigrationRunner} from "../../core/runnerExecution";
import {assertIdleHealth} from "../../core/runnerGuest";
const id="11111111-1111-4111-8111-111111111111";
test("target location accepts ARM display names but rejects other or missing regions",()=>{
  for(const location of ["Japan East","japaneast","JAPANEAST"])assert.equal(sameAzureLocation(location,"japaneast"),true);
  for(const location of ["Japan West","",undefined,null,{},19])assert.equal(sameAzureLocation(location,"japaneast"),false);
});
function fixture(){
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});r.phase="provisioned";
  r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
  const requests:string[]=[];let result:unknown;
  const control:RunnerControl={persist:async()=>{},sleep:async()=>{},list:async()=>[],request:async(_s,_p,method="GET")=>{requests.push(method);return {status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify(result)}}}};}};
  return {r,control,requests,result:(x:unknown)=>{result=x;}};
}
test("migration cannot start without complete target, sizing and fresh safe Linux health",async()=>{
  const f=fixture();await assert.rejects(migrationPreflight(f.control,f.r,"{}"),/target/);assert.deepEqual(f.requests,[]);
  const ready={bootId:id,cliVersion:"dev",archiveSha256:f.r.artifact.sha256,commit:"a".repeat(40),checkedAt:new Date().toISOString(),health:{idle:true,storageUsedPercent:6,swapUsedBytes:0,oomEvents:0}};
  f.r.guestReady=ready;assertIdleHealth(f.r);
  for(const health of [undefined,{...ready.health,idle:false},{...ready.health,storageUsedPercent:80},{...ready.health,swapUsedBytes:1},{...ready.health,oomEvents:1}])assert.throws(()=>assertIdleHealth({...f.r,guestReady:{...ready,health}}));
  assert.throws(()=>assertIdleHealth({...f.r,guestReady:{...ready,checkedAt:"2020-01-01T00:00:00Z"}}));
});
test("migration receipt must match retained job, boot, action and fingerprint before terminal success",async()=>{
  const f=fixture();f.r.migration={operation:id,jobId:id,phase:"submitted",bootId:id,startedAt:new Date().toISOString(),artifactSHA256:f.r.artifact.sha256,cliVersion:"dev",evidence:{} as any};
  f.r.guestCommand={id:f.r.vmId+"/runCommands/af-"+id,operation:id,action:"migrate-csv",phase:"submitted",submittedAt:new Date().toISOString()};
  const receipt={version:1,workflow:id,operation:id,jobId:id,action:"migrate-csv",bootId:id,configSha256:"b".repeat(64),phase:"finished",exitCode:0,reportBytes:300,reportSha256:"c".repeat(64),fingerprint:"d".repeat(64)};
  f.result(receipt);const r=await refreshMigration(f.control,f.r);assert.equal(r.migration?.phase,"finished");assert.equal(r.migration?.verification,undefined);assert.deepEqual(f.requests,["GET"]);
  for(const patch of [{jobId:"foreign"},{bootId:"foreign"},{action:"inventory"},{fingerprint:"invalid"},{reportBytes:0},{exitCode:1}]){f.result({...receipt,...patch});await assert.rejects(refreshMigration(f.control,f.r));}
});
test("target DSN is host-bound TLS with password URL encoded and absent from retained metadata",()=>{
  const f=fixture();f.r.target={input:{serverName:"afpg-test"}} as any;
  const secret="a".repeat(24)+"/@:?&";
  const dsn=targetDSN(f.r,secret),u=new URL(dsn);
  assert.equal(u.hostname,"afpg-test.postgres.database.azure.com");assert.equal(decodeURIComponent(u.password),secret);assert.equal(u.search,"?sslmode=verify-full");assert.equal(u.pathname,"/agefreighter");assert.ok(!JSON.stringify(f.r).includes(secret));
});

function settlingRunner(){
  const f=fixture();
  f.r.target={input:{loaderSize:"Standard_D4s_v5",deadline:new Date(Date.now()+3600000).toISOString(),budgetUSD:800,additionalReserveUSD:400,hourlyUSD:0.736}} as any;
  f.r.guestReady={bootId:id,cliVersion:"dev",archiveSha256:f.r.artifact.sha256,commit:"a".repeat(40),checkedAt:new Date().toISOString(),health:{idle:true,storageUsedPercent:6,swapUsedBytes:0,oomEvents:0}};
  let reads=0,sleeps=0;let states=["Updating","Succeeded"];
  const vm={tags:{workflow:id},properties:{hardwareProfile:{vmSize:"Standard_D4s_v5"},instanceView:{statuses:[{code:"PowerState/running"}]}}};
  f.control.request=async(_s,_p,method="GET")=>{assert.equal(method,"GET");const state=states[Math.min(reads++,states.length-1)];return {status:200,value:{...vm,properties:{...vm.properties,provisioningState:state}}};};
  f.control.sleep=async(ms)=>{assert.equal(ms,2000);sleeps++;};
  f.control.persist=async()=>{assert.fail("read-only wait must not persist or submit a job");};
  return {...f,vm,states:(value:string[])=>{states=value;},counts:()=>({reads,sleeps})};
}
test("migration waits read-only for running matching VM ARM update to settle",async()=>{
  const f=settlingRunner();await waitForMigrationRunner(f.control,f.r);assert.deepEqual(f.counts(),{reads:2,sleeps:1});
});
test("migration ARM wait is bounded and rejects unrelated state changes",async()=>{
  const f=settlingRunner();f.states(["Updating"]);await assert.rejects(waitForMigrationRunner(f.control,f.r),/30 seconds/);assert.deepEqual(f.counts(),{reads:16,sleeps:15});
  for(const state of ["Failed","Creating","Deleting",""]){const g=settlingRunner();g.states([state]);await assert.rejects(waitForMigrationRunner(g.control,g.r),/provisioning state/);assert.equal(g.counts().reads,1);}
  for(const field of ["workflow","size","power"]){const g=settlingRunner();if(field==="workflow")g.vm.tags.workflow="foreign";if(field==="size")g.vm.properties.hardwareProfile.vmSize="Standard_B2s_v2";if(field==="power")g.vm.properties.instanceView.statuses[0]!.code="PowerState/stopped";await assert.rejects(waitForMigrationRunner(g.control,g.r),/changed/);assert.equal(g.counts().sleeps,0);}
});
test("migration ARM wait rechecks health and budget after waiting",async()=>{
  for(const field of ["health","budget"]){const f=settlingRunner();f.control.sleep=async()=>{if(field==="health")f.r.guestReady!.health!.oomEvents=1;else f.r.target!.input.deadline="2020-01-01T00:00:00Z";};await assert.rejects(waitForMigrationRunner(f.control,f.r));assert.equal(f.counts().reads,1);}
});
