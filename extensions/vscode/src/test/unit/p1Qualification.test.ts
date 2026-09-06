import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {verifyP1,p1Script,p1ExportScript,P1Qualification} from "../../core/p1Qualification";
import {sourceWorkflowDraft} from "../../core/runner";
const id="11111111-1111-4111-8111-111111111111";
function report(){const expected=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/p1-canonical-expected-20260906.json","utf8"));return {version:1,jobId:id,readOnly:true,expected,actual:{...structuredClone(expected),source:"apache-age",jobId:id},comparison:{status:"pass"}};}
test("P1 verifier recomputes all 64 canonical leaves and cannot trust a forged summary",()=>{
  verifyP1(JSON.stringify(report()),id);
  for(const change of [(r:any)=>r.actual.leaves.pop(),(r:any)=>r.actual.leaves[0].sha256="a".repeat(64),(r:any)=>r.actual.leaves[0].rows++,(r:any)=>r.actual.jobId="foreign",(r:any)=>r.expected.rootSha256="a".repeat(64),(r:any)=>r.readOnly=false,(r:any)=>r.actual.leaves[0].name+="\0",(r:any)=>r.comparison.status="fail"]){const r=report();change(r);assert.throws(()=>verifyP1(JSON.stringify(r),id));}
});
test("P1 execution and just-in-time export are separate and preserve the pinned loader",()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  r.migration={jobId:id} as any;r.target={input:{serverName:"afpg-test"}} as any;
  const q:P1Qualification={operation:id,commandId:r.vmId+"/runCommands/af-"+id,jobId:id,artifact:{version:"dev",sha256:"a".repeat(64),url:`https://af${id.replaceAll("-","").slice(0,22)}.blob.core.windows.net/af-${id}/artifacts/${"a".repeat(64)}.tar.gz`,development:{commit:"b".repeat(40),bytes:100}},startedAt:new Date().toISOString(),phase:"submitted"};
  q.artifact.version="2.4.0-dev.bbbbbbbbbbbb";
  const run=p1Script(r,q);
  assert.ok(run.includes("MemoryMax=4G"));assert.ok(run.includes("MemorySwapMax=0"));
  assert.ok(!run.includes("AF_P1_REPORT"));assert.ok(!run.includes("/usr/local/bin"));
  assert.throws(()=>p1ExportScript(r,q));
  const transfer=p1ExportScript(r,{...q,sha256:"c".repeat(64),bytes:400});
  assert.ok(!transfer.includes("AF_P1_DSN"));assert.ok(!transfer.includes("systemd-run"));
  assert.ok(transfer.includes("If-None-Match"));assert.ok(transfer.includes("hashlib.sha256(data).hexdigest()=='"+"c".repeat(64)));
});
