import assert from "node:assert/strict";
import test from "node:test";
import {diagnosticGate,diagnosticReceipt,p1DiagnosticScript,P1Diagnostic} from "../../core/p1Diagnostic";
import {sourceWorkflowDraft} from "../../core/runner";
import {requalificationGate,p1Script} from "../../core/p1Qualification";
const id="11111111-1111-4111-8111-111111111111",op="22222222-2222-4222-8222-222222222222";
function fixture(){
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  r.artifact={version:"2.4.0-dev.bbbbbbbbbbbb",sha256:"a".repeat(64),url:`https://af${id.replaceAll("-","").slice(0,22)}.blob.core.windows.net/af-${id}/artifacts/${"a".repeat(64)}.tar.gz`,development:{commit:"b".repeat(40),bytes:100}};
  r.migration={jobId:id,phase:"finished",verification:{outcome:"pass"}} as any;
  r.p1Qualification={operation:id,jobId:id,phase:"failed"} as any;
  r.target={phase:"provisioned",input:{serverName:"afpg-test",deadline:new Date(Date.now()+3600000).toISOString(),budgetUSD:800,additionalReserveUSD:400,hourlyUSD:0.736}} as any;
  r.guestReady={bootId:id,checkedAt:new Date().toISOString(),cliVersion:r.artifact.version,archiveSha256:r.artifact.sha256,commit:r.artifact.development!.commit,health:{idle:false,storageUsedPercent:6,swapUsedBytes:0,oomEvents:0}};
  const d:P1Diagnostic={operation:op,commandId:r.vmId+"/runCommands/af-"+op,jobId:id,failedOperation:id,phase:"submitted",startedAt:new Date().toISOString(),artifact:r.artifact};return {r,d};
}
test("diagnostic requires failed qualifier, complete counts, fresh pinned health and budget",()=>{
  const {r}=fixture();assert.doesNotThrow(()=>diagnosticGate(r));
  for(const change of [(x:any)=>x.migration.verification.outcome="incomplete",(x:any)=>x.p1Qualification.phase="submitted",(x:any)=>x.guestReady.checkedAt="2000-01-01",(x:any)=>x.guestReady.health.swapUsedBytes=1,(x:any)=>x.guestReady.health.oomEvents=1,(x:any)=>x.guestReady.health.storageUsedPercent=80,(x:any)=>x.guestReady.archiveSha256="changed",(x:any)=>x.target.input.deadline="2000-01-01"]){const bad=structuredClone(r);change(bad);assert.throws(()=>diagnosticGate(bad));}
});
test("legacy diagnostics cannot silently fall back for a retained Gremlin qualification",()=>{
  const {r,d}=fixture();r.p1Qualification!.profile="gremlin-partition64";
  assert.throws(()=>diagnosticGate(r),/profile-specific/);
  assert.throws(()=>p1DiagnosticScript(r,d),/Gremlin profile/);
  assert.throws(()=>requalificationGate(r));
});
test("diagnosis is separate, protected, non-replaying and preserves active failure marker",()=>{
  const {r,d}=fixture(),s=p1DiagnosticScript(r,d);
  assert.match(s,/flock -n 9/);assert.match(s,/pgrep -x agefreighter/);assert.match(s,/pgrep -x p1runnerverify/);
  assert.match(s,/MemoryMax=4G/);assert.match(s,/MemorySwapMax=0/);assert.match(s,/failure\.json/);
  assert.ok(!s.includes('rm '));assert.ok(!s.includes(' load '));assert.ok(!s.includes('resume'));assert.ok(!s.includes('> "$root/active"'));
  assert.throws(()=>p1DiagnosticScript(r,{...d,operation:id}));
  assert.throws(()=>p1DiagnosticScript(r,{...d,failedOperation:op}));
});
test("diagnostic receipt cannot assert qualification success or leak unrecognized fields",()=>{
  const {r,d}=fixture();const v={workflow:id,operation:op,jobId:id,failedOperation:id,readOnly:true,exitCode:1,bytes:95,sha256:"a".repeat(64),failure:{version:1,outcome:"fail",stage:"target-digest",code:"source-key-order",secret:"not allowed"},secret:"not allowed"};
  const out=diagnosticReceipt(v,r,d);assert.ok(!JSON.stringify(out).includes("not allowed"));
  for(const bad of [{...v,jobId:op},{...v,readOnly:false},{...v,exitCode:0},{...v,bytes:4000},{...v,failure:{...v.failure,code:"postgresql://secret"}},{...v,failure:{...v.failure,outcome:"pass"}}])assert.throws(()=>diagnosticReceipt(bad,r,d));
});
test("explicit ordering correction preserves failure history and refuses unrelated retries",()=>{
  const {r,d}=fixture();
  assert.throws(()=>requalificationGate(r));
  r.p1Diagnostic={...d,phase:"finished",result:{failure:{stage:"target-digest",code:"source-key-order"}}};
  assert.doesNotThrow(()=>requalificationGate(r));
  const bad=structuredClone(r);bad.p1Diagnostic!.result!.failure={stage:"comparison",code:"canonical-mismatch"};assert.throws(()=>requalificationGate(bad));
  const q={...r.p1Qualification!,operation:op,artifact:r.artifact,replacesFailedOperation:id};
  assert.throws(()=>p1Script(r,q));
  r.p1QualificationHistory=[structuredClone(r.p1Qualification!)];
  const script=p1Script(r,q);assert.match(script,/active\.retained/);assert.match(script,/mv "\$root\/active"/);assert.match(script,/pgrep -x p1runnerverify/);
  assert.ok(!script.includes(" load "));assert.equal(r.p1QualificationHistory[0]!.phase,"failed");
});
