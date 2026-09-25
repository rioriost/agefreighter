import assert from "node:assert/strict";
import test from "node:test";
import {mkdtemp, readFile, rm, stat} from "node:fs/promises";
import {spawnSync} from "node:child_process";
import {join} from "node:path";
import {tmpdir} from "node:os";
import {sourceWorkflowDraft} from "../../core/runner";
import {RunnerControl} from "../../core/runnerLifecycle";
import {refreshMigration} from "../../core/runnerExecution";
import {RunnerLockedError, RunnerStore} from "../../guided/runnerStore";

const id="11111111-1111-4111-8111-111111111111";
const operation="22222222-2222-4222-8222-222222222222";
function retained(){
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",size:"Standard_D4s_v5",subnetId:"unused",source:{type:"neo4j",location:"on-premises"}});
  r.phase="provisioned";
  r.artifact={version:"test",sha256:"a".repeat(64),url:"https://example.invalid/archive"};
  r.migration={operation,jobId:operation,phase:"submitted",bootId:id,startedAt:new Date().toISOString(),artifactSHA256:r.artifact.sha256,cliVersion:"test",evidence:{operation,reportSHA256:"b".repeat(64),configurationSHA256:"c".repeat(64),artifactSHA256:r.artifact.sha256,sourceType:"neo4j",rows:"5600000",vertices:"1600000",edges:"4000000",storageHighBytes:"100000000",labels:{}}};
  r.guestCommand={id:r.vmId+"/runCommands/af-"+operation,operation,action:"migrate-source",phase:"unknown",submittedAt:new Date().toISOString()};
  return r;
}

test("reopening a persisted uncertain migration is inert and reconciles without another write request",async()=>{
  const root=await mkdtemp(join(tmpdir(),"af-reconnect-"));
  try{
    const first=new RunnerStore(root),r=retained();await first.write(r);
    const bytes=await readFile(join(root,id+".json"));
    const reopened=new RunnerStore(root);
    assert.deepEqual(await reopened.list(),[r]);
    assert.deepEqual(await readFile(join(root,id+".json")),bytes);
    const methods:string[]=[];
    const control:RunnerControl={sleep:async()=>{},list:async()=>{throw new Error("unexpected list");},persist:r=>reopened.write(r),request:async(_s,_p,method="GET")=>{
      methods.push(method);return {status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify({version:1,workflow:id,operation,jobId:operation,action:"migrate-source",bootId:id,configSha256:"c".repeat(64),phase:"running"})}}}};
    }};
    const next=await reopened.exclusive(id,async()=>refreshMigration(control,await reopened.read(id)));
    assert.deepEqual(methods,["GET"]);assert.equal(next.migration?.jobId,operation);
    assert.equal(next.migration?.phase,"running");assert.equal(next.migration?.verification,undefined);
    assert.deepEqual(await new RunnerStore(root).read(id),next);
  }finally{await rm(root,{recursive:true,force:true});}
});

test("a terminated lock owner remains blocked after reconnect and cannot silently repair or replay",async()=>{
  const root=await mkdtemp(join(tmpdir(),"af-crash-lock-"));
  try{
    const store=new RunnerStore(root),r=retained();await store.write(r);
    const lock=join(root,id+".lock"),before=await readFile(join(root,id+".json"));
    // A separate process exits without the controller's finally/unlink path.
    const child=spawnSync(process.execPath,["-e","require('node:fs').openSync(process.argv[1],'wx',0o600);process.exit(23)",lock]);
    assert.equal(child.status,23);let actions=0;
    const reopened=new RunnerStore(root);
    await assert.rejects(reopened.exclusive(id,async()=>{actions++;}),RunnerLockedError);
    assert.equal(actions,0);assert((await stat(lock)).isFile());
    assert.deepEqual(await readFile(join(root,id+".json")),before);
  }finally{await rm(root,{recursive:true,force:true});}
});

for(const phase of ["failed","interrupted"] as const){
  test(`reconnect retains ${phase} state without creating a replacement job or verification pass`,async()=>{
    const r=retained(),methods:string[]=[];
    const control:RunnerControl={sleep:async()=>{},list:async()=>[],persist:async()=>{},request:async(_s,_p,method="GET")=>{
      methods.push(method);return {status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify({version:1,workflow:id,operation,jobId:operation,action:"migrate-source",bootId:id,configSha256:"c".repeat(64),phase})}}}};
    }};
    const next=await refreshMigration(control,r);
    assert.equal(next.migration?.phase,phase);assert.equal(next.migration?.jobId,operation);
    assert.equal(next.migration?.verification,undefined);assert.deepEqual(methods,["GET"]);
  });
}
