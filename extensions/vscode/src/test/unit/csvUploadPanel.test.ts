import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {createRequire} from "node:module";
import {join} from "node:path";
import {Script} from "node:vm";
import {transformSync} from "esbuild";
import {object, sourceWorkflowDraft} from "../../core/runner";
import {CSVManifest, CSVTransferCancelledError} from "../../guided/csvTransfer";

// Production panel handler with inert dialogs, store, and Azure adapters.
// This is not installed-GUI or real Blob failure evidence.
const id="11111111-1111-4111-8111-111111111111";
const code=transformSync(readFileSync(join(__dirname,"../../runnerSourcePanel.ts"),"utf8"),{loader:"ts",format:"cjs"}).code;
function fixture(acknowledged=false){
 let record=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",size:"Standard_B2s_v2",subnetId:"unused",source:{type:"csv",location:"local"}});
 record.storageDeployment={phase:"ready"} as typeof record.storageDeployment;
 record.sourceFiles=[1,2,3].map(n=>({id:`00000000-0000-4000-8000-${String(n).padStart(12,"0")}`,name:`${n}.csv`,path:`/test/${n}.csv`}));
 let receive=async(_m:unknown)=>{},cancelUpload=true,changed=false,confirmations=0,disposedListeners=0;
 let cancel=()=>{};
 const messages:Record<string,any>[]=[],uploaded:string[]=[],snapshots:typeof record[]=[];
 const store={read:async()=>structuredClone(record),write:async(r:typeof record)=>{record=structuredClone(r);snapshots.push(structuredClone(r));},exclusive:async(_id:string,action:()=>unknown)=>action()};
 const control={persist:store.write};
 const modules:Record<string,unknown>={
  vscode:{ViewColumn:{One:1},ProgressLocation:{Notification:15},workspace:{isTrusted:true},window:{
   createWebviewPanel:()=>({onDidDispose:()=>{},webview:{html:"",postMessage:async(m:Record<string,any>)=>{messages.push(m);},onDidReceiveMessage:(fn:typeof receive)=>{receive=fn;return {dispose:()=>{}};}}}),
   showWarningMessage:async()=>{confirmations++;return "Upload reviewed CSV files";},
   withProgress:async(options:{cancellable:boolean},action:(p:unknown,t:unknown)=>unknown)=>{
    let canceled=false;
    const token={get isCancellationRequested(){return canceled;},onCancellationRequested:(fn:()=>void)=>{cancel=()=>{canceled=true;fn();};return {dispose:()=>{disposedListeners++;}};}};
    return action({report:()=>{}},options.cancellable?token:undefined);
   }
  }},
  "./core/runner":{object},
  "./core/runnerAssessment":{assessmentActive:()=>false},
  "./core/runnerSourceView":{runnerSourceHTML:()=>"test"},
  "./core/runnerReportStorage":{reportStorageNames:()=>({origin:"https://test.invalid"}),verifyTransferStorage:async()=>{}},
  "./guided/csvTransfer":{CSVTransferCancelledError,inspectCSV:async(file:string)=>({file,bytes:10,sha256:(changed?"b":"a").repeat(64)})}
 };
 const output={exports:{openRunnerSource:(_context:unknown,_control:unknown,_store:unknown,_id:string,_services:unknown)=>{}}},native=createRequire(__filename);
 new Script(code).runInNewContext({module:output,exports:output.exports,Error,AbortController,require:(name:string)=>name in modules?modules[name]:name.startsWith("node:")?native(name):{}});
 output.exports.openRunnerSource({subscriptions:[]},control,store,id,{uploadCSV:async(_record:unknown,_path:string,manifest:CSVManifest,_progress:unknown,signal:AbortSignal)=>{
  uploaded.push(manifest.file);
  if(cancelUpload&&manifest.file===record.sourceFiles![1]!.id){cancel();if(!acknowledged)throw new CSVTransferCancelledError();}
  assert.equal(signal.aborted,acknowledged&&cancelUpload&&manifest.file===record.sourceFiles![1]!.id);
 }});
 return {run:()=>receive({action:"uploadCSV"}),record:()=>record,messages,uploaded,snapshots,confirmations:()=>confirmations,disposed:()=>disposedListeners,retry:()=>{cancelUpload=false;},change:()=>{changed=true;}};
}

for(const acknowledged of [false,true])test(`CSV batch cancellation preserves prepared state and does not attempt later files (acknowledged=${acknowledged})`,async()=>{
 const f=fixture(acknowledged);await f.run();
 const r=f.record(),files=r.sourceFiles!;
 assert.deepEqual(r.csvTransfers?.map(t=>t.phase),["uploaded","prepared"]);
 assert.deepEqual(f.uploaded,[files[0]!.id,files[1]!.id]);
 assert.ok(f.messages.some(m=>m.kind==="error"&&/canceled/.test(m.text)));
 assert.equal(f.confirmations(),1);assert.equal(f.disposed(),2);
 f.retry();await f.run();
 assert.equal(f.confirmations(),2); // retry must pass the approval surface again
 assert.deepEqual(f.uploaded,[files[0]!.id,files[1]!.id,files[1]!.id,files[2]!.id]);
 assert.deepEqual(f.record().csvTransfers?.map(t=>t.phase),["uploaded","uploaded","uploaded"]);
 assert.equal(f.record().assessment,undefined);assert.equal(f.record().migration,undefined);
});

test("a changed manifest on explicit retry cannot overwrite the retained batch",async()=>{
 const f=fixture();await f.run();const before=structuredClone(f.record()),calls=f.uploaded.length;
 f.retry();f.change();await f.run();
 assert.deepEqual(f.record(),before);assert.equal(f.uploaded.length,calls);
 assert.ok(f.messages.some(m=>m.kind==="error"&&/previously reviewed CSV changed/.test(m.text)));
});
