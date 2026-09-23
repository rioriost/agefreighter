import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {verifyP1,p1Script,p1ExportScript,assertP1Projection,parseP1Receipt,P1Qualification,p1ProfileForConfiguration,p1ProfileSpec,assertP1VerifierManifest,p1FixtureRoot,P1RejectionError} from "../../core/p1Qualification";
import {sourceWorkflowDraft} from "../../core/runner";
import {buildSourceDraft} from "../../core/runnerSource";
import {sourceForm,workflow} from "../sourceFixtures";
const id="11111111-1111-4111-8111-111111111111";
test("terminal P1 command errors without a JSON object are admitted to failure reconciliation",()=>{
  for(const value of [undefined,null,"", "Independent qualification failed; inspect retained evidence\n","null","[]","1",'"error"'])assert.deepEqual(parseP1Receipt(value),{});
  assert.deepEqual(parseP1Receipt('{"verified":true,"jobId":"job"}'),{verified:true,jobId:"job"});
});
function report(){const expected=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/p1-canonical-expected-20260906.json","utf8"));return {version:1,jobId:id,readOnly:true,expected,actual:{...structuredClone(expected),source:"apache-age",jobId:id},comparison:{status:"pass"}};}
const gremlinConfig={source:{type:"cosmos-nosql",cosmos:{gremlin:{enabled:true,partitionKeyProperty:"partitionKey",propertyTypes:{score:"float64",distance_km:"float64"}}}}};
test("Gremlin P1 requires a separately reviewed profile, typed projection and verifier artifact",()=>{
  assert.equal(p1ProfileForConfiguration(gremlinConfig),"gremlin-partition64");assert.equal(p1ProfileForConfiguration(undefined),"raw-id");
  assertP1Projection(gremlinConfig);
  for(const field of ["score","distance_km"]){const c=structuredClone(gremlinConfig);delete (c.source.cosmos.gremlin.propertyTypes as any)[field];assert.throws(()=>assertP1Projection(c));}
  const spec=p1ProfileSpec("gremlin-partition64"),m={purpose:"p1-read-only-verifier",fixtureRoot:p1FixtureRoot,canonicalRoot:spec.root,canonicalVersion:spec.version,qualificationProfile:"gremlin-partition64",archive:"verifier.tar.gz"};
  assertP1VerifierManifest(m,"gremlin-partition64");assert.throws(()=>assertP1VerifierManifest(m,"raw-id"));
  for(const field of ["qualificationProfile","canonicalVersion","canonicalRoot"]){const bad:any={...m};delete bad[field];assert.throws(()=>assertP1VerifierManifest(bad,"gremlin-partition64"));}
});
test("Gremlin report validates every partition-preserving range and rejects offline or raw-ID substitutions",()=>{
  // Synthetic target-role envelope for parser testing, not live target evidence.
  const offline=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/gremlin-offline-p1-20260917.json","utf8"));
  const d={version:1,jobId:id,readOnly:true,qualificationProfile:"gremlin-partition64",expected:offline.expected,actual:{...offline.actual,source:"apache-age",jobId:id},comparison:{status:"pass"}};
  verifyP1(JSON.stringify(d),id,"gremlin-partition64");assert.throws(()=>verifyP1(JSON.stringify(d),id));
  for(const change of [(r:any)=>r.actual.source="cosmos-gremlin-offline",(r:any)=>delete r.qualificationProfile,(r:any)=>r.actual.leaves[0].sha256="0".repeat(64),(r:any)=>r.actual.leaves.reverse(),(r:any)=>r.actual.canonicalVersion="agefreighter-production-simulation-v1"]){const bad=structuredClone(d);change(bad);assert.throws(()=>verifyP1(JSON.stringify(bad),id,"gremlin-partition64"));}
  assert.throws(()=>verifyP1(JSON.stringify(report()),id,"gremlin-partition64"));
});
function projection(type="postgresql"):any{
  const vertices="Supplier Facility Product PurchaseOrder Shipment Lot Location Carrier Customer".split(" ").map(label=>({label,idField:"external_id",properties:Object.fromEntries("source_key external_id name region created_at status score active tags quantities description".split(" ").map(p=>[p,p]))}));
  const edges="SUPPLIES PRODUCED_AT PLACED_WITH CONTAINS FULFILLS ORIGINATES_AT DESTINED_FOR CARRIED_BY INCLUDED_IN".split(" ").map(label=>({label,externalIdField:"relationship_id",properties:Object.fromEntries("source_key relationship_id occurred_at quantity status distance_km notes".split(" ").map(p=>[p,p]))}));
  return {source:{type,[type==="cosmos-nosql"?"cosmos":type]:{vertices,edges}}};
}
test("P1 rejects the AZ-PGVM projection omission even when identity fields exist",()=>{
  const c=projection();const pg=c.source.postgresql!;
  for(const row of pg.vertices){delete row.properties.source_key;delete row.properties.external_id;}
  for(const row of pg.edges){delete row.properties.source_key;delete row.properties.relationship_id;}
  assert.throws(()=>assertP1Projection(c),/Supplier.*source_key, external_id/);
});
test("P1 projection admission accepts full explicit mappings but not missing, duplicated or extra properties",()=>{
  for(const type of ["postgresql","csv","cosmos-nosql"])assert.doesNotThrow(()=>assertP1Projection(projection(type)));
  for(const change of [(p:any)=>p.vertices.pop(),(p:any)=>p.vertices[1].label="Supplier",(p:any)=>delete p.edges[0].properties.relationship_id,(p:any)=>p.vertices[0].properties.extra="extra"]){const c=projection();change(c.source.postgresql);assert.throws(()=>assertP1Projection(c));}
  assert.throws(()=>assertP1Projection({}));
  assert.doesNotThrow(()=>assertP1Projection({source:{type:"neo4j"}}));
});
test("corrected reviewed PostgreSQL P1 mappings include identity properties and source keys in every generated query",()=>{
  const mappings=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/fixtures/postgresql-p1-mappings.json","utf8"));
  const draft=buildSourceDraft({type:"postgresql",location:"azure"},{...sourceForm,port:5432,mappings},workflow);
  assert.doesNotThrow(()=>assertP1Projection(draft.configuration));
  const pg=(draft.configuration.source as any).postgresql;
  for(const row of [...pg.vertices,...pg.edges]){assert.match(row.query,/"source_key"/);assert.equal(row.properties.source_key,"source_key");}
  assert.ok(!draft.warnings.some(w=>w.includes("stable ID field is used for identity only")));
});
test("P1 verifier recomputes all 64 canonical leaves and cannot trust a forged summary",()=>{
  verifyP1(JSON.stringify(report()),id);
  for(const change of [(r:any)=>r.actual.leaves.pop(),(r:any)=>r.actual.leaves[0].sha256="a".repeat(64),(r:any)=>r.actual.leaves[0].rows++,(r:any)=>r.actual.jobId="foreign",(r:any)=>r.expected.rootSha256="a".repeat(64),(r:any)=>r.readOnly=false,(r:any)=>r.actual.leaves[0].name+="\0",(r:any)=>r.comparison.status="fail"]){const r=report();change(r);assert.throws(()=>verifyP1(JSON.stringify(r),id));}
});
test("P1 structural rejection uses fixed categories without exposing malformed report contents",()=>{
  for(const text of ['{"private-secret":',"[]","null",JSON.stringify({...report(),comparison:[]})]){
    assert.throws(()=>verifyP1(text,id),error=>error instanceof P1RejectionError&&error.category==="json-shape"&&!String(error.stack).includes("private-secret")&&!("cause" in error));
  }
});
test("P1 execution and just-in-time export are separate and preserve the pinned loader",()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  r.migration={jobId:id} as any;r.target={input:{serverName:"afpg-test"}} as any;
  const q:P1Qualification={operation:id,commandId:r.vmId+"/runCommands/af-"+id,jobId:id,artifact:{version:"dev",sha256:"a".repeat(64),url:`https://af${id.replaceAll("-","").slice(0,22)}.blob.core.windows.net/af-${id}/artifacts/${"a".repeat(64)}.tar.gz`,development:{commit:"b".repeat(40),bytes:100}},startedAt:new Date().toISOString(),phase:"submitted"};
  q.artifact.version="2.4.0-dev.bbbbbbbbbbbb";
  const run=p1Script(r,q);
  r.sourceDraft={configuration:gremlinConfig} as any;
  assert.throws(()=>p1Script(r,q),/profile differs/);
  const gremlinRun=p1Script(r,{...q,profile:"gremlin-partition64"});assert.match(gremlinRun,/'gremlin-partition64'/);
  r.sourceDraft=undefined;
  assert.ok(run.includes("MemoryMax=4G"));assert.ok(run.includes("MemorySwapMax=0"));
  assert.ok(!run.includes("AF_P1_REPORT"));assert.ok(!run.includes("/usr/local/bin"));
  assert.throws(()=>p1ExportScript(r,q));
  const transfer=p1ExportScript(r,{...q,sha256:"c".repeat(64),bytes:400});
  assert.ok(!transfer.includes("AF_P1_DSN"));assert.ok(!transfer.includes("systemd-run"));
  assert.ok(transfer.includes("If-None-Match"));assert.ok(transfer.includes("hashlib.sha256(data).hexdigest()=='"+"c".repeat(64)));
});
