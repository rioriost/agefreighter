import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import * as runner from "../../core/runner";
import * as catalog from "../../core/runnerCatalog";
import * as assessment from "../../core/runnerAssessment";
import * as source from "../../core/runnerSource";
import { catalogFixture, catalogForm, catalogSHA, catalogText } from "../catalogFixtures";
import { sourceForm, workflow } from "../sourceFixtures";

const code=transformSync(readFileSync(join(__dirname,"../../runnerSourcePanel.ts"),"utf8"),{loader:"ts",format:"cjs"}).code;
function fixture(){
  const f=catalogFixture(),workspace={isTrusted:true},messages:any[]=[],confirmations:string[]=[],writes:runner.RunnerRecord[]=[];
  let receive=async(_m:unknown)=>{},dispose=()=>{},onConfirm=()=>{},onPassword=()=>{},cancel=false,passwords=0;
  const store={read:async()=>structuredClone(f.record),write:async(r:runner.RunnerRecord)=>{f.record=structuredClone(r);writes.push(r);},exclusive:async(_id:string,fn:()=>unknown)=>fn(),readReport:async()=>catalogText};
  const modules:Record<string,unknown>={vscode:{workspace,ViewColumn:{One:1},window:{
    createWebviewPanel:()=>({onDidDispose:(fn:()=>void)=>{dispose=fn;},webview:{html:"",postMessage:async(m:unknown)=>{messages.push(m);},onDidReceiveMessage:(fn:typeof receive)=>{receive=fn;return{dispose:()=>{}};}}}),
    showWarningMessage:async(title:string,_options:unknown,button:string)=>{confirmations.push(title);onConfirm();return cancel?undefined:button;},
    showInputBox:async()=>{passwords++;onPassword();return "PRIVATE-CATALOG-PASSWORD";}
  }},"./core/runner":runner,"./core/runnerCatalog":catalog,"./core/runnerAssessment":assessment,"./core/runnerSource":source,"./core/runnerSourceView":{runnerSourceHTML:()=>"fixture"}};
  const output={exports:{openRunnerSource:(_a:unknown,_b:unknown,_c:unknown,_d:string)=>{}}},native=createRequire(__filename);
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,Buffer,require:(name:string)=>name in modules?modules[name]:name.startsWith("node:")?native(name):{}});
  output.exports.openRunnerSource({subscriptions:[]},f.control,store,workflow);
  return {f,workspace,messages,writes,confirmations,run:(m:unknown)=>receive(m),cancel:()=>{cancel=true;},dispose:()=>dispose(),onConfirm:(fn:()=>void)=>{onConfirm=fn;},onPassword:(fn:()=>void)=>{onPassword=fn;},passwords:()=>passwords};
}
const start={action:"catalogStart",form:catalogForm,schemas:["public"]};
test("catalog native approval and password dispatch once without retaining credentials or mappings",async()=>{
  const p=fixture();await p.run(start);
  assert.equal(p.passwords(),1);assert.equal(p.confirmations.length,1);assert.equal(p.f.requests.filter(x=>x.method==="PUT").length,1);
  assert.equal(p.f.record.sourceDraft,undefined);assert.equal(p.f.record.postgresCatalog?.phase,"submitted");
  assert.ok(!JSON.stringify(p.f.saved).includes("PRIVATE-CATALOG-PASSWORD"));assert.ok(!JSON.stringify(p.messages).includes("PRIVATE-CATALOG-PASSWORD"));
  await p.run(start);assert.equal(p.f.requests.filter(x=>x.method==="PUT").length,1);
});
for(const change of ["untrusted","old-guest","cancel","dispose","trust","boot","source","artifact","mapping"]){
  test(`catalog approval rejects ${change} before source dispatch`,async()=>{
    const p=fixture();
    if(change==="untrusted")p.workspace.isTrusted=false;
    if(change==="old-guest")p.f.record.guestReady!.capabilities=[];
    if(change==="cancel")p.cancel();
    p.onConfirm(()=>{if(change==="dispose")p.dispose();});
    p.onPassword(()=>{
      if(change==="trust")p.workspace.isTrusted=false;
      if(change==="boot")p.f.record.guestReady!.bootId="22222222-2222-4222-8222-222222222222";
      if(change==="source")p.f.record.input.source.location="other-cloud";
      if(change==="artifact")p.f.record.artifact.sha256="b".repeat(64);
      if(change==="mapping")p.f.record.sourceDraft=source.buildSourceDraft(p.f.record.input.source,sourceForm,workflow);
    });
    await p.run(start);assert.equal(p.f.requests.length,0);assert.equal(p.f.record.postgresCatalog,undefined);
  });
}

async function finished(p:ReturnType<typeof fixture>){
  await p.run(start);const c=p.f.record.postgresCatalog!;
  p.f.record.postgresCatalog={...c,phase:"finished",reportSHA256:catalogSHA,reportBytes:Buffer.byteLength(catalogText)};
  p.f.record.guestCommand!.phase="finished";
  p.f.record.reportTransfers=[{operation:c.operation,sha256:catalogSHA,bytes:Buffer.byteLength(catalogText),blob:"fixture",phase:"imported"}];
  return catalog.catalogRecommendations(p.f.record,catalogText).proposals[0]!.id;
}
test("catalog adoption requires native confirmation, keeps manual mappings and invalidates in-window review",async()=>{
  const p=fixture(),id=await finished(p),form={...catalogForm,mappings:[sourceForm.mappings[0]!]};
  const calls=p.f.requests.length;
  await p.run({action:"catalogAdopt",form,schemas:["public"],selected:[id]});
  assert.equal(p.writes.length,1);assert.equal(p.f.record.sourceDraft?.form.mappings.length,2);assert.deepEqual(p.f.record.sourceDraft?.form.mappings[0],form.mappings[0]);
  assert.equal(p.f.requests.length,calls);assert.ok(p.messages.some(x=>x.kind==="catalogAdopted"));
  await p.run({action:"assess",method:"inventory"});assert.equal(p.f.requests.length,calls);assert.match(p.messages.filter(x=>x.kind==="error").at(-1).text,/Review current source/);
});
for(const change of ["cancel","trust","mapping","target","catalog"]){test(`catalog adoption refuses ${change} without overwriting`,async()=>{
  const p=fixture(),id=await finished(p);
  p.onConfirm(()=>{
    if(change==="cancel")p.cancel();if(change==="trust")p.workspace.isTrusted=false;
    if(change==="mapping")p.f.record.sourceDraft=source.buildSourceDraft(p.f.record.input.source,sourceForm,workflow);
    if(change==="target")p.f.record.target={} as any;
    if(change==="catalog")p.f.record.postgresCatalog!.reportSHA256="b".repeat(64);
  });
  await p.run({action:"catalogAdopt",form:catalogForm,schemas:["public"],selected:[id]});assert.equal(p.writes.length,0);
});}
