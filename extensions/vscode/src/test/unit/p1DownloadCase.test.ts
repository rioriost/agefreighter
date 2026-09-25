import assert from "node:assert/strict";
import test,{describe,TestContext} from "node:test";
import {createHash,randomBytes} from "node:crypto";
import {chmod,link,lstat,mkdir,mkdtemp,readFile,readdir,rm,symlink,writeFile} from "node:fs/promises";
import {join,resolve} from "node:path";
import {tmpdir} from "node:os";
import {execFile} from "node:child_process";
import {promisify} from "node:util";
import {sourceWorkflowDraft,RunnerRecord} from "../../core/runner";
import {maxReportBytes} from "../../core/runnerBlob";
import {RunnerStore} from "../../guided/runnerStore";
import {prepareP1DownloadCase,disposableCaseParent,DownloadCaseInput} from "../helpers/prepareP1DownloadCase";

const id="11111111-1111-4111-8111-111111111111",exportOperation="22222222-2222-4222-8222-222222222222";
const hash=(data:Uint8Array|string)=>createHash("sha256").update(data).digest("hex");
function record():RunnerRecord{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/test/subnets/runner`,size:"Standard_B2s_v2",source:{type:"neo4j",location:"other-cloud"}});
  r.phase="provisioned";r.artifact={version:"2.4.0",sha256:"a".repeat(64),url:"https://example.invalid/runner.tar.gz"};
  r.target={phase:"provisioned",input:{serverName:"afpg-test"},serverId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.DBforPostgreSQL/flexibleServers/afpg-test`} as RunnerRecord["target"];
  r.sourceDraft={configuration:{source:{type:"neo4j",neo4j:{password:{env:"AGEFREIGHTER_SOURCE_PASSWORD"}}}}} as unknown as RunnerRecord["sourceDraft"];
  r.migration={phase:"finished",jobId:id,verification:{outcome:"pass",summary:"synthetic test only"}} as RunnerRecord["migration"];
  r.p1Qualification={operation:id,jobId:id,phase:"pass",commandId:`${r.vmId}/runCommands/af-${id}`,exportCommandId:`${r.vmId}/runCommands/af-${exportOperation}`,artifact:r.artifact,startedAt:"2026-09-23T00:00:00Z",sha256:"c".repeat(64),bytes:512};
  return r;
}
async function fixture(t:TestContext,change:(r:RunnerRecord)=>void=()=>{}){
  const parent=await disposableCaseParent(),inputs=await mkdtemp(join(parent,"af-b12-input-")),caseRoot=join(parent,`af-b12-${randomBytes(6).toString("hex")}`);
  t.after(async()=>{
    // Only synthetic directories created by this test. Production preparation
    // deliberately retains partial evidence and has no cleanup operation.
    try{await chmod(join(caseRoot,"originals"),0o700);}catch{}
    await rm(inputs,{recursive:true,force:true});await rm(caseRoot,{recursive:true,force:true});
  });
  const r=record();change(r);const q=r.p1Qualification!;
  const receipt={workflow:r.id,operation:q.operation,jobId:q.jobId,sha256:q.sha256,bytes:q.bytes,exported:true};
  const recordPath=join(inputs,"authorized-snapshot.json"),manifestPath=join(inputs,"guest-export-receipt.json");
  const original=Buffer.from(JSON.stringify(r,null,2)+"\n"),manifest=Buffer.from(JSON.stringify(receipt,null,2)+"\n");
  await writeFile(recordPath,original,{flag:"wx",mode:0o600});await writeFile(manifestPath,manifest,{flag:"wx",mode:0o600});
  const options:DownloadCaseInput={recordPath,manifestPath,caseRoot,scenario:"wrong-sha256",acknowledgeNonsecretInputs:true};
  return {options,r,receipt,original,manifest,inputs};
}
async function absent(path:string){await assert.rejects(lstat(path),error=>(error as NodeJS.ErrnoException).code==="ENOENT");}

// Every case below creates POSIX-only private fixture directories. Preserve the
// complete success and negative contract on supported hosts, not a Windows mock.
describe("POSIX offline download fixture", {skip: process.platform === "win32"}, () => {
for(const phase of ["exported","pass"] as const)for(const scenario of ["wrong-sha256","wrong-length"] as const)test(`offline ${scenario} case from ${phase} changes only declared expectations`,async t=>{
  const f=await fixture(t,r=>{r.p1Qualification!.phase=phase;});f.options.scenario=scenario;
  const result=await prepareP1DownloadCase(f.options),storage=join(result.userDataDir,"User","globalStorage","rioriost.agefreighter","runner-v2");
  const actual=await new RunnerStore(storage).read(id),expected=structuredClone(f.r);expected.p1Qualification!.phase="exported";
  if(scenario==="wrong-sha256")expected.p1Qualification!.sha256="0"+"c".repeat(63);else expected.p1Qualification!.bytes=513;
  assert.deepEqual(actual,expected);assert.deepEqual(await readdir(storage),[`${id}.json`]);
  assert.deepEqual(await readFile(join(result.caseRoot,"originals","record.json")),f.original);
  assert.deepEqual(await readFile(join(result.caseRoot,"originals","guest-export-receipt.json")),f.manifest);
  assert.deepEqual(await readFile(f.options.recordPath),f.original);assert.deepEqual(await readFile(f.options.manifestPath),f.manifest);
  assert.equal((await lstat(result.caseRoot)).mode&0o777,0o700);assert.equal((await lstat(join(result.caseRoot,"originals","record.json"))).mode&0o777,0o400);
  const evidence=JSON.parse(await readFile(result.evidencePath,"utf8"));
  assert.equal(evidence.originalRecordSHA256,hash(f.original));assert.equal(evidence.originalReceiptSHA256,hash(f.manifest));
  assert.equal(evidence.originalSeal.sha256,f.receipt.sha256);assert.equal(evidence.originalSeal.bytes,512);
  assert.equal(evidence.disposableRecordSHA256,hash(await readFile(result.recordPath)));
  assert.equal(evidence.reportsCopied,0);assert.equal(evidence.networkRequests,0);assert.equal(evidence.qualification,"not-run");
  assert.equal(evidence.changes.length,phase==="pass"?2:1);assert.deepEqual(await readdir(result.extensionsDir),[]);
  await assert.rejects(prepareP1DownloadCase(f.options),/already exist/);
  assert.deepEqual(await new RunnerStore(storage).read(id),expected);
});
test("length corruption remains within the production bound at maximum length",async t=>{
  const f=await fixture(t,r=>{r.p1Qualification!.bytes=maxReportBytes;});f.options.scenario="wrong-length";
  const result=await prepareP1DownloadCase(f.options),actual=JSON.parse(await readFile(result.recordPath,"utf8"));
  assert.equal(actual.p1Qualification.bytes,maxReportBytes-1);
});

const invalidRecords:[string,(r:any)=>void][]=[
  ["foreign workflow schema",r=>{r.schemaVersion=1;}],
  ["unprovisioned workflow",r=>{r.phase="draft";}],
  ["invalid workflow UUID",r=>{r.id="bad";}],
  ["changed VM",r=>{r.vmId+="-changed";}],
  ["changed deployment",r=>{r.deploymentId+="-changed";}],
  ["missing target",r=>{delete r.target;}],
  ["unprovisioned target",r=>{r.target.phase="previewed";}],
  ["target outside subscription",r=>{r.target.serverId=r.target.serverId.replace(id,exportOperation);}],
  ["unfinished migration",r=>{r.migration.phase="submitted";}],
  ["failed counts",r=>{r.migration.verification.outcome="fail";}],
  ["unexported qualification",r=>{r.p1Qualification.phase="verified";}],
  ["missing export command",r=>{delete r.p1Qualification.exportCommandId;}],
  ["reused export command",r=>{r.p1Qualification.exportCommandId=r.p1Qualification.commandId;}],
  ["wrong guest command",r=>{r.p1Qualification.commandId+="-changed";}],
  ["wrong job",r=>{r.p1Qualification.jobId=exportOperation;}],
  ["changed source type",r=>{r.sourceDraft.configuration.source.type="csv";}],
  ["changed profile",r=>{r.p1Qualification.profile="gremlin-partition64";}],
  ["unknown profile",r=>{r.p1Qualification.profile="future";}],
  ["unknown record field",r=>{r.extra={value:"unreviewed"};}],
  ["unknown qualification field",r=>{r.p1Qualification.extra=true;}],
  ["nested password",r=>{r.sourceDraft.configuration.source.neo4j.password="sensitive";}],
  ["nested bearer token",r=>{r.template.authorization="Bearer sensitive";}],
  ["inline token value",r=>{r.template.description="Bearer sensitive";}],
  ["SAS URL",r=>{r.artifact.url+="?sig=sensitive";}],
  ["private key",r=>{r.template.description="-----BEGIN PRIVATE KEY-----";}],
  ["credential DSN",r=>{r.template.description="postgresql://user:private@host/db";}]
];
for(const [name,change] of invalidRecords)test(`offline fixture refuses ${name} before retaining any raw bytes`,async t=>{
  const f=await fixture(t,change);await assert.rejects(prepareP1DownloadCase(f.options));await absent(f.options.caseRoot);
  assert.deepEqual(await readFile(f.options.recordPath),f.original);
});
for(const field of ["workflow","operation","jobId","sha256","bytes","exported","unknown"] as const)test(`independent guest receipt must bind ${field}`,async t=>{
  const f=await fixture(t);const receipt:any={...f.receipt};receipt[field]=field==="bytes"?513:field==="exported"?false:"different";
  await writeFile(f.options.manifestPath,JSON.stringify(receipt));await assert.rejects(prepareP1DownloadCase(f.options));await absent(f.options.caseRoot);
});
test("known nonsecret ARM parameter declarations remain admissible, literal parameter values do not",async t=>{
  const f=await fixture(t,r=>{r.template={parameters:{administratorPassword:{type:"secureString"}},properties:{administratorLoginPassword:"[parameters('administratorPassword')]",disablePasswordAuthentication:true,passwordAuth:"Enabled"}};});
  await prepareP1DownloadCase(f.options);
  const bad=await fixture(t,r=>{r.template={parameters:{administratorPassword:{type:"secureString",defaultValue:"private"}}};});
  await assert.rejects(prepareP1DownloadCase(bad.options));await absent(bad.options.caseRoot);
});
test("retained native resize and cost approval histories are preserved with narrowly validated shapes",async t=>{
  const f=await fixture(t,r=>{
    r.resizeAuthorization={binding:"b".repeat(64),approvedAt:"2026-09-22T00:00:00.000Z",deadline:"2026-09-22T00:20:00.000Z"};
    const cost={deadline:"2026-09-22T02:00:00.000Z",budgetUSD:800,additionalReserveUSD:500,hourlyUSD:1};
    r.costAuthorizations=[{authorizedAt:"2026-09-22T00:00:00.000Z",previous:cost,current:{...cost,deadline:"2026-09-22T03:00:00.000Z"}}];
  });
  const result=await prepareP1DownloadCase(f.options),copy=JSON.parse(await readFile(result.recordPath,"utf8"));
  assert.deepEqual(copy.resizeAuthorization,f.r.resizeAuthorization);assert.deepEqual(copy.costAuthorizations,f.r.costAuthorizations);
});
for(const bad of [
  {resizeAuthorization:{binding:"Bearer secret",approvedAt:"2026-09-22T00:00:00Z",deadline:"2026-09-22T00:20:00Z"}},
  {resizeAuthorization:{binding:"a".repeat(64),approvedAt:"2026-09-22T00:00:00Z",deadline:"2026-09-22T00:20:00Z",unreviewed:"secret"}},
  {costAuthorizations:[{authorizedAt:"2026-09-22T00:00:00Z",previous:{password:"secret"},current:{}}]}
])test(`approval audit allowance rejects unknown or secret data (${hash(JSON.stringify(bad)).slice(0,8)})`,async t=>{
  const f=await fixture(t,r=>Object.assign(r,bad));await assert.rejects(prepareP1DownloadCase(f.options));await absent(f.options.caseRoot);
});
test("Gremlin profile with production default-Azure credential reference is preserved",async t=>{
  const f=await fixture(t,r=>{
    r.input.source={type:"cosmos-nosql",location:"azure",resourceId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.DocumentDB/databaseAccounts/test`};
    r.sourceDraft!.configuration={source:{type:"cosmos-nosql",cosmos:{credential:"default-azure",gremlin:{enabled:true}}}};
    r.p1Qualification!.profile="gremlin-partition64";
  });
  const result=await prepareP1DownloadCase(f.options),copy=JSON.parse(await readFile(result.recordPath,"utf8"));
  assert.deepEqual(copy.sourceDraft,f.r.sourceDraft);assert.equal(copy.p1Qualification.profile,"gremlin-partition64");
});
test("malformed UTF-8 and oversized snapshots fail before output",async t=>{
  const f=await fixture(t);await writeFile(f.options.recordPath,Buffer.from([0xff]));
  await assert.rejects(prepareP1DownloadCase(f.options));await absent(f.options.caseRoot);
  await writeFile(f.options.recordPath,Buffer.alloc(2*1024*1024+1));
  await assert.rejects(prepareP1DownloadCase(f.options),/bounded/);await absent(f.options.caseRoot);
});
for(const raw of ['{"schemaVersion":2,"schemaVersion":2}', '{"schemaVersion":2,"nested":{"password":"secret","password":{"env":"AGEFREIGHTER_SOURCE_PASSWORD"}}}', "[]", "null", "{", '{"schemaVersion":2,"__proto__":{}}'])test(`ambiguous or invalid JSON is refused before output (${hash(raw).slice(0,8)})`,async t=>{
  const f=await fixture(t);await writeFile(f.options.recordPath,raw);await assert.rejects(prepareP1DownloadCase(f.options));await absent(f.options.caseRoot);
});
test("direct operator storage inputs are refused before filesystem access",async t=>{
  const f=await fixture(t);f.options.recordPath="/does-not-exist/User/globalStorage/rioriost.agefreighter/runner-v2/snapshot.json";
  await assert.rejects(prepareP1DownloadCase(f.options),/Operator storage/);await absent(f.options.caseRoot);
});
test("symlink and hardlinked input files are refused",async t=>{
  const f=await fixture(t),linked=join(f.inputs,"linked.json");await symlink(f.options.recordPath,linked);
  await assert.rejects(prepareP1DownloadCase({...f.options,recordPath:linked}),/Symbolic links/);await absent(f.options.caseRoot);
  const hard=join(f.inputs,"hard.json");await link(f.options.recordPath,hard);
  await assert.rejects(prepareP1DownloadCase({...f.options,recordPath:hard}),/hard links/);await absent(f.options.caseRoot);
});
test("symlink input ancestor and symlink case root cannot redirect setup",async t=>{
  const f=await fixture(t),alias=join(f.inputs,"alias");await symlink(f.inputs,alias);
  await assert.rejects(prepareP1DownloadCase({...f.options,recordPath:join(alias,"authorized-snapshot.json")}),/Symbolic links/);
  await symlink(f.inputs,f.options.caseRoot);await assert.rejects(prepareP1DownloadCase(f.options),/already exist/);
  assert.deepEqual(await readFile(f.options.recordPath),f.original);
});
test("existing empty or nonempty case directories and arbitrary destinations are refused",async t=>{
  const f=await fixture(t);await mkdir(f.options.caseRoot,{mode:0o700});await assert.rejects(prepareP1DownloadCase(f.options),/already exist/);
  const sentinel=join(f.options.caseRoot,"preserve.txt");await writeFile(sentinel,"preserve");await assert.rejects(prepareP1DownloadCase(f.options),/already exist/);
  assert.equal(await readFile(sentinel,"utf8"),"preserve");
  await assert.rejects(prepareP1DownloadCase({...f.options,caseRoot:join(f.inputs,"af-b12-000000000000")}),/fresh case root/);
});
test("simultaneous setup of one case has exactly one winner and preserves the original evidence",async t=>{
  const f=await fixture(t),results=await Promise.allSettled([prepareP1DownloadCase(f.options),prepareP1DownloadCase(f.options)]);
  assert.equal(results.filter(r=>r.status==="fulfilled").length,1);assert.equal(results.filter(r=>r.status==="rejected").length,1);
  assert.deepEqual(await readFile(join(f.options.caseRoot,"originals","record.json")),f.original);
});
test("acknowledgment, supported scenario and distinct input paths are required",async t=>{
  const f=await fixture(t);
  await assert.rejects(prepareP1DownloadCase({...f.options,acknowledgeNonsecretInputs:false} as unknown as DownloadCaseInput));
  await assert.rejects(prepareP1DownloadCase({...f.options,scenario:"pass"} as unknown as DownloadCaseInput));
  await assert.rejects(prepareP1DownloadCase({...f.options,manifestPath:f.options.recordPath}),/separate/);await absent(f.options.caseRoot);
});
test("test-only CLI prepares the explicit disposable case and prints paths without snapshot contents",async t=>{
  const f=await fixture(t),helper=resolve(__dirname,"../helpers/prepareP1DownloadCase.ts");
  const result=await promisify(execFile)(process.execPath,[require.resolve("tsx/cli"),helper,"--record",f.options.recordPath,"--manifest",f.options.manifestPath,"--case-root",f.options.caseRoot,"--scenario","wrong-length","--acknowledge-nonsecret-inputs"],{timeout:10000,maxBuffer:4096});
  assert.equal(result.stderr,"");const output=JSON.parse(result.stdout);
  assert.equal(output.caseRoot,f.options.caseRoot);assert.equal("originalSeal" in output,false);assert.equal("record" in output,false);
  assert.equal(JSON.parse(await readFile(output.recordPath,"utf8")).p1Qualification.bytes,513);
});
test("CLI argument failure is sanitized and does not create a case",async t=>{
  const f=await fixture(t),helper=resolve(__dirname,"../helpers/prepareP1DownloadCase.ts");
  await assert.rejects(promisify(execFile)(process.execPath,[require.resolve("tsx/cli"),helper,"--unknown","private-not-for-output"],{timeout:10000,maxBuffer:4096}),error=>{
    const result=error as Error&{stderr:string;stdout:string};assert.equal(result.stdout,"");
    assert.match(result.stderr,/B12 setup refused/);assert.ok(!result.stderr.includes("private-not-for-output"));assert.ok(!result.stderr.includes(" at "));return true;
  });await absent(f.options.caseRoot);
});

});

test("Windows refuses offline fixture preparation before reading inputs or creating a case", {skip: process.platform !== "win32"}, async t => {
  const root = await mkdtemp(join(tmpdir(), "af-b12-unsupported-"));
  t.after(() => rm(root, {recursive: true, force: true}));
  const options: DownloadCaseInput = {recordPath: join(root, "missing-record.json"), manifestPath: join(root, "missing-receipt.json"),
    caseRoot: join(root, "af-b12-000000000000"), scenario: "wrong-length", acknowledgeNonsecretInputs: true};
  await assert.rejects(disposableCaseParent(), /requires POSIX private directories/);
  await assert.rejects(prepareP1DownloadCase(options), /requires POSIX private directories/);
  assert.deepEqual(await readdir(root), []);
  const helper = resolve(__dirname, "../helpers/prepareP1DownloadCase.ts");
  await assert.rejects(promisify(execFile)(process.execPath, [require.resolve("tsx/cli"), helper,
    "--record", options.recordPath, "--manifest", options.manifestPath, "--case-root", options.caseRoot,
    "--scenario", options.scenario, "--acknowledge-nonsecret-inputs"], {timeout: 10000, maxBuffer: 4096}), error => {
    const result = error as Error & {stderr: string; stdout: string};
    assert.equal(result.stdout, ""); assert.match(result.stderr, /B12 setup refused.*requires POSIX private directories/);
    assert.ok(!result.stderr.includes(options.recordPath)); assert.ok(!result.stderr.includes(" at ")); return true;
  });
  assert.deepEqual(await readdir(root), []);
});
