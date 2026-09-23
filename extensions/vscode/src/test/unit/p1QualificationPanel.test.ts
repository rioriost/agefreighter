import assert from "node:assert/strict";
import test from "node:test";
import {createHash} from "node:crypto";
import {readFileSync} from "node:fs";
import {createRequire} from "node:module";
import {join} from "node:path";
import {Script} from "node:vm";
import {transformSync} from "esbuild";
import {sourceWorkflowDraft, RunnerRecord} from "../../core/runner";
import * as qualification from "../../core/p1Qualification";
import * as blob from "../../core/runnerBlob";
import {escapeHTML} from "../../core/report";

// Actual production controller, parser, manifest and download verification.
// Dialog/store/ARM/HTTP adapters are inert: not signed-in GUI or Azure evidence.
const id="11111111-1111-4111-8111-111111111111";
const foreign="22222222-2222-4222-8222-222222222222";
const code=transformSync(readFileSync(join(__dirname,"../../p1QualificationPanel.ts"),"utf8"),{loader:"ts",format:"cjs"}).code;
const hash=(text:string)=>createHash("sha256").update(text).digest("hex");
function report():any {
  const expected=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/p1-canonical-expected-20260906.json","utf8"));
  return {version:1,jobId:id,readOnly:true,expected,actual:{...structuredClone(expected),source:"apache-age",jobId:id},comparison:{status:"pass"}};
}
function fixture(change:(value:any)=>void=()=>{}, transport="valid") {
  const doc=report();change(doc);
  const text=transport==="truncated"?JSON.stringify(doc).slice(0,-1):JSON.stringify(doc);
  let record=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"unused",size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
  record.migration={jobId:id,phase:"finished",verification:{outcome:"pass",summary:"counts only"}} as RunnerRecord["migration"];
  record.p1Qualification={operation:id,jobId:id,commandId:`${record.vmId}/runCommands/af-${id}`,startedAt:new Date().toISOString(),phase:"exported",artifact:record.artifact,sha256:transport==="hash"?"0".repeat(64):hash(text),bytes:Buffer.byteLength(text)+(transport==="length"?1:0)};
  const snapshots:RunnerRecord[]=[],panels:{title:string;options:any;webview:{html:string}}[]=[],requests:string[]=[],retained:string[]=[];
  const store={read:async()=>structuredClone(record),exclusive:async(_id:string,fn:()=>unknown)=>fn(),
    retainReport:async(_id:string,m:blob.ReportManifest,value:string)=>{blob.verifyReportBytes(Buffer.from(value),m);retained.push(value);},
    readReport:async(_id:string,m:blob.ReportManifest)=>blob.verifyReportBytes(Buffer.from(text),m)};
  const control={persist:async(r:RunnerRecord)=>{record=structuredClone(r);snapshots.push(structuredClone(r));},request:async()=>{throw Error("Unexpected ARM operation");},list:async()=>{throw Error("Unexpected ARM list");}};
  const capability=()=>{
    const now=Date.now(),q=new URLSearchParams({sv:"2023-11-03",spr:"https",sr:"b",sp:"r",st:new Date(now-1000).toISOString(),se:new Date(now+600000).toISOString(),sig:"synthetic",skoid:id,sktid:id,skt:new Date(now-2000).toISOString(),ske:new Date(now+700000).toISOString(),sks:"b",skv:"2023-11-03"});
    return `https://af${id.replaceAll("-","").slice(0,22)}.blob.core.windows.net/af-${id}/reports/${id}.json?${q}`;
  };
  const modules:Record<string,unknown>={
    vscode:{ViewColumn:{Beside:2},workspace:{isTrusted:true},window:{createWebviewPanel:(_kind:string,title:string,_column:number,options:unknown)=>{const panel={title,options,webview:{html:""}};panels.push(panel);return panel;}}},
    "./developmentRunner":{developmentEnabled:()=>true},
    "./core/p1Qualification":qualification,
    "./core/runnerReportStorage":{verifyTransferStorage:async()=>{}},
    "./core/report":{escapeHTML},
    "./core/runnerBlob":{...blob,downloadReport:(raw:string,workflow:string,m:blob.ReportManifest)=>blob.downloadReport(raw,workflow,m,async(_url,init)=>{
      requests.push(init!.method!);
      if(transport==="throw")throw Error("synthetic private URL must not escape");
      return new Response(text,{status:transport==="403"?403:200});
    })}
  };
  const output={exports:{qualifyP1:async(..._args:unknown[]):Promise<RunnerRecord>=>record}},native=createRequire(__filename);
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,Buffer,require:(name:string)=>name in modules?modules[name]:name.startsWith("node:")?native(name):{}});
  return {run:()=>output.exports.qualifyP1({},control,store,{reportCapability:async()=>capability()},id),record:()=>record,snapshots,panels,requests,retained,text};
}

test("P1 controller imports exact complete report before persisting and displaying PASS; reopening performs no download",async()=>{
  const f=fixture();await f.run();
  assert.equal(f.record().p1Qualification?.phase,"pass");assert.deepEqual(f.requests,["GET"]);
  assert.equal(f.retained.length,1);assert.equal(f.panels.length,1);
  assert.equal(f.panels[0]!.title,"Verified P1 migration");
  assert.equal(f.panels[0]!.options.enableScripts,false);assert.equal(f.panels[0]!.options.localResourceRoots.length,0);
  assert.match(f.panels[0]!.webview.html,/P1 full canonical digest: PASS/);
  await f.run();assert.deepEqual(f.requests,["GET"]);assert.equal(f.panels.length,2);
});

const invalidReports:[string,(r:any)=>void,qualification.P1RejectionCategory][]=[
  ["foreign envelope job",r=>{r.jobId=foreign;},"identity-outcome"],
  ["foreign target job",r=>{r.actual.jobId=foreign;},"identity-outcome"],
  ["missing range",r=>{r.actual.leaves.pop();},"coverage"],
  ["duplicated range",r=>{r.actual.leaves[1]=r.actual.leaves[0];},"canonical-root"],
  ["reordered ranges",r=>{r.actual.leaves.reverse();},"canonical-root"],
  ["forged matching summaries and leaves",r=>{r.expected.leaves[0].sha256="a".repeat(64);r.actual.leaves[0].sha256="a".repeat(64);},"canonical-root"],
  ["wrong fixture root",r=>{r.actual.fixtureRootSha256="a".repeat(64);},"coverage"],
  ["unknown canonical version",r=>{r.actual.canonicalVersion="future";},"coverage"],
  ["different record count",r=>{r.actual.recordCount--;},"coverage"],
  ["comparison failure",r=>{r.comparison.status="fail";},"identity-outcome"],
  ["non-read-only receipt",r=>{r.readOnly=false;},"identity-outcome"],
  ["missing expected object",r=>{delete r.expected;},"json-shape"],
  ["array actual object",r=>{r.actual=[];},"json-shape"],
  ["null comparison object",r=>{r.comparison=null;},"json-shape"],
  ["null range",r=>{r.actual.leaves[0]=null;},"range"],
  ["unsafe integer range",r=>{r.actual.leaves[0].startKey=Number.MAX_SAFE_INTEGER+1;},"range"],
  ["foreign report profile",r=>{r.qualificationProfile="gremlin-partition64";},"profile"]
];
for(const [name,change,category] of invalidReports)test(`P1 controller never persists or displays PASS for ${name}`,async()=>{
  const f=fixture(change),before=structuredClone(f.record());
  await assert.rejects(f.run(),error=>{
    assert.ok(error instanceof qualification.P1RejectedImportError);assert.equal(error.category,category);
    assert.deepEqual(error.evidence,{status:"rejected",retained:true,manifest:{operation:id,sha256:hash(f.text),bytes:Buffer.byteLength(f.text)},jobId:id,profile:"raw-id"});
    assert.ok(Object.isFrozen(error.evidence));assert.ok(Object.isFrozen(error.evidence.manifest));
    assert.match(error.message,/Rejected evidence is retained; it is not an accepted P1 result/);
    assert.ok(!JSON.stringify(error).includes("synthetic"));return true;
  });
  assert.deepEqual(f.record(),before);assert.equal(f.snapshots.length,0);assert.equal(f.panels.length,0);
  assert.deepEqual(f.requests,["GET"]);
  // Hash-valid invalid documents remain available as failed qualification evidence.
  assert.deepEqual(f.retained,[f.text]);
});
for(const kind of ["hash","length","truncated","403","throw"])test(`P1 controller refuses ${kind} transfer before retention or success`,async()=>{
  const f=fixture(()=>{},kind),before=structuredClone(f.record());
  const categories:Record<string,blob.ReportRejectionCategory>={hash:"sha256",length:"length",truncated:"json","403":"http-status",throw:"transport"};
  await assert.rejects(f.run(),error=>error instanceof blob.ReportRejectionError&&error.category===categories[kind]&&/Report download could not be verified/.test(error.message));
  assert.deepEqual(f.record(),before);assert.equal(f.snapshots.length,0);assert.equal(f.panels.length,0);assert.equal(f.retained.length,0);assert.deepEqual(f.requests,["GET"]);
});

test("reopening an invalid historical pass preserves immutable evidence without promoting or redownloading",async()=>{
  const f=fixture(r=>{r.actual.jobId=foreign;});f.record().p1Qualification!.phase="pass";
  const before=structuredClone(f.record());
  await assert.rejects(f.run(),error=>error instanceof qualification.P1RejectedImportError&&error.category==="identity-outcome"&&error.evidence.manifest.sha256===hash(f.text));
  assert.deepEqual(f.record(),before);assert.equal(f.panels.length,0);assert.equal(f.snapshots.length,0);
  assert.deepEqual(f.requests,[]);assert.deepEqual(f.retained,[f.text]);
});

for(const field of ["jobId","commandId"] as const)test(`P1 controller rejects changed retained ${field} before any download`,async()=>{
  const f=fixture();f.record().p1Qualification![field]=foreign;
  const before=structuredClone(f.record());await assert.rejects(f.run(),/identity changed/);
  assert.deepEqual(f.record(),before);assert.equal(f.panels.length,0);assert.equal(f.snapshots.length,0);assert.deepEqual(f.requests,[]);
});

test("reopening previously passing but hash-corrupted retained bytes cannot display PASS or redownload",async()=>{
  const f=fixture(()=>{},"hash");f.record().p1Qualification!.phase="pass";
  const before=structuredClone(f.record());await assert.rejects(f.run(),/length or SHA-256/);
  assert.deepEqual(f.record(),before);assert.equal(f.panels.length,0);assert.equal(f.snapshots.length,0);assert.equal(f.retained.length,0);assert.deepEqual(f.requests,[]);
});

test("P1 controller rejects changed Gremlin source/profile before transfer or mutation",async()=>{
  const f=fixture();f.record().sourceDraft={configuration:{source:{type:"cosmos-nosql",cosmos:{gremlin:{enabled:true}}}}} as any;
  const before=structuredClone(f.record());await assert.rejects(f.run(),/profile differs/);
  assert.deepEqual(f.record(),before);assert.deepEqual(f.requests,[]);assert.equal(f.panels.length,0);assert.equal(f.snapshots.length,0);
});

test("P1 controller imports only the bound Gremlin profile and reopens without redownload",async()=>{
  // The controller is real; transport/target envelope is a local synthetic test.
  const offline=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/gremlin-offline-p1-20260917.json","utf8"));
  const f=fixture(d=>{d.qualificationProfile="gremlin-partition64";d.expected=offline.expected;d.actual={...offline.actual,source:"apache-age",jobId:id};});
  f.record().sourceDraft={configuration:{source:{type:"cosmos-nosql",cosmos:{gremlin:{enabled:true}}}}} as any;
  f.record().p1Qualification!.profile="gremlin-partition64";
  await f.run();assert.equal(f.record().p1Qualification?.phase,"pass");assert.equal(f.panels.length,1);assert.deepEqual(f.requests,["GET"]);
  await f.run();assert.deepEqual(f.requests,["GET"]);assert.equal(f.panels.length,2);
});
