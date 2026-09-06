import assert from "node:assert/strict";
import test from "node:test";
import {sourceWorkflowDraft} from "../../core/runner";
import {RunnerControl} from "../../core/runnerLifecycle";
import {migrationPreflight,refreshMigration,targetDSN,sameAzureLocation} from "../../core/runnerExecution";
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
