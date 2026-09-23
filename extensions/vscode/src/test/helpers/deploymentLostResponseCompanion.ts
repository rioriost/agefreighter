/** Build-only; never launches VS Code, signs in, installs or calls Azure. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, lstat, realpath, mkdir, open, readdir } from "node:fs/promises";
import { join } from "node:path";
import { build } from "esbuild";
import { LostResponseScope } from "./deploymentLostResponseStages";
import { retainLostResponseFile } from "./deploymentLostResponse";
import { sourceWorkflowDraft, parseRunnerInput, RunnerRecord } from "../../core/runner";
import { developmentArtifact } from "../../core/runnerDevelopment";
import { validateLostResponseSeed } from "./deploymentLostResponseSeed";
const sha=(v:Uint8Array)=>createHash("sha256").update(v).digest("hex");
export async function prepareLostResponseCompanion(root:string,extensionRoot:string,scope:LostResponseScope,continuationRoot?:string):Promise<string[]> {
  assert.match(root,/^\/private\/tmp\/af-deployment-lost-response-[a-zA-Z0-9]{6}$/); assert.equal(await realpath(root),root); assert.equal((await lstat(root)).mode&0o777,0o700);
  assert.deepEqual(parseRunnerInput(scope.input),scope.input); assert.deepEqual(scope.input.source,{type:"csv",location:"local"});
  assert.ok(scope.annotation.includes("agefreighter-tools")&&scope.annotation.includes("negative")); assert.ok(Number.isFinite(Date.parse(scope.expiresAt)));
  const manifest=JSON.parse(await readFile(scope.manifestPath,"utf8")),artifact=developmentArtifact(sourceWorkflowDraft(scope.workflow,scope.input),manifest);
  assert.equal(artifact.sha256,scope.artifactSHA256);assert.equal(artifact.development!.bytes,scope.artifactBytes);
  const archive=await readFile(scope.archivePath);assert.equal(archive.length,scope.artifactBytes);assert.equal(sha(archive),scope.artifactSHA256);
  assert.equal(join(scope.manifestPath.slice(0,scope.manifestPath.lastIndexOf('/')),manifest.archive),scope.archivePath);
  let seed=Buffer.from(JSON.stringify(sourceWorkflowDraft(scope.workflow,scope.input)));
  let continuation:unknown;
  if(continuationRoot){
    assert.match(continuationRoot,/^\/private\/tmp\/af-deployment-lost-response-[a-zA-Z0-9]{6}$/);
    assert.equal(await realpath(continuationRoot),continuationRoot);
    assert.equal((await lstat(continuationRoot)).mode&0o777,0o700);
    assert.equal(sha(await readFile(join(continuationRoot,"scope.json"))),sha(Buffer.from(JSON.stringify(scope,null,2)+"\n")));
    const ledger=await readdir(join(continuationRoot,"ledger"));assert.ok(ledger.every(n=>/^stage-\d{4}\.json$/.test(n)),"No effect intent may precede continuation");
    for(const name of ledger){const e=JSON.parse(await readFile(join(continuationRoot,"ledger",name),"utf8"));assert.ok(["subscriptions","locations","storagePrincipal","LIST","GET","POST-whatIf","whatIf-poll-header"].includes(e.kind));}
    const sourceFile=join(continuationRoot,"runner-v2",scope.workflow+".json");const info=await lstat(sourceFile);assert.ok(info.isFile()&&!info.isSymbolicLink()&&info.size<262144);assert.equal(info.mode&0o777,0o600);
    seed=await readFile(sourceFile);validateLostResponseSeed(JSON.parse(seed.toString()) as RunnerRecord,scope);
    continuation={root:continuationRoot,recordSHA256:sha(seed),ledgerFiles:ledger,cloudEffectsObserved:false,storageRoleWillBeReviewedAgain:true};
  }
  const initialRecordSHA256=sha(seed);
  await retainLostResponseFile(root,"preparing.json",{at:new Date().toISOString(),cloudExecuted:false});
  const source=join(extensionRoot,"src"),runtime=join(source,"test/helpers/deploymentLostResponseRuntime.ts");
  const result=await build({entryPoints:[runtime],bundle:true,platform:"node",format:"cjs",target:"node22",write:false,metafile:true,logLevel:"silent",external:["vscode"],plugins:[{name:"b09-instance-facade",setup(builder){
    builder.onResolve({filter:/^vscode$/},args=>args.kind==="import-statement"?{path:"vscode",namespace:"b09"}:undefined);
    builder.onResolve({filter:/\/guided\/azure$/},args=>args.importer!==runtime?{path:"azure",namespace:"b09"}:undefined);
    builder.onLoad({filter:/.*/,namespace:"b09"},args=>({loader:"js",resolveDir:source,contents:args.path==="azure"?`export class AzureSession {constructor(){return globalThis.__afLostResponse.azure;}}`:
      `const actual=module.require("vscode"); const ns=name=>new Proxy({}, {get:(_,key)=>(globalThis.__afLostResponse?.vscode??actual)[name][key]}); export const authentication=ns("authentication"),env=ns("env"),commands=ns("commands"),window=ns("window"),workspace=ns("workspace"),Uri=ns("Uri"),ViewColumn=ns("ViewColumn"),ProgressLocation=ns("ProgressLocation"); export const Disposable=actual.Disposable,EventEmitter=actual.EventEmitter,CancellationTokenSource=actual.CancellationTokenSource;`}));
  }}]});assert.equal(result.outputFiles.length,1);
  const write=async(name:string,value:string|Uint8Array)=>{const f=await open(join(root,name),"wx",0o600);try{await f.writeFile(value);await f.sync();}finally{await f.close();}};
  await write("initial-record.json",seed);
  const scopeSHA256=sha(Buffer.from(JSON.stringify(scope,null,2)+"\n"));
  const bundle=result.outputFiles[0]!.contents,bundleSHA256=sha(bundle);await write("runtime.cjs",bundle);
  await write("extension.cjs",`const fs=require("node:fs/promises"),crypto=require("node:crypto");exports.activate=async context=>{const bytes=await fs.readFile(__dirname+"/runtime.cjs");if(crypto.createHash("sha256").update(bytes).digest("hex")!==${JSON.stringify(bundleSHA256)})throw Error("B09 bundle changed");const scope=await fs.readFile(__dirname+"/scope.json");if(crypto.createHash("sha256").update(scope).digest("hex")!==${JSON.stringify(scopeSHA256)})throw Error("B09 scope changed");const seed=await fs.readFile(__dirname+"/initial-record.json");if(crypto.createHash("sha256").update(seed).digest("hex")!==${JSON.stringify(initialRecordSHA256)})throw Error("B09 initial record changed");return require("./runtime.cjs").activate(context);};`);
  await retainLostResponseFile(root,"scope.json",scope);
  await retainLostResponseFile(root,"package.json",{name:"agefreighter-b09-lost-response",publisher:"agefreighter-test-only",version:"0.0.0",displayName:"B09 disposable negative fixture",engines:{vscode:"^1.105.0"},main:"./extension.cjs",activationEvents:["onCommand:agefreighterB09.open","onCommand:agefreighterB09.artifact"],extensionKind:["ui"],contributes:{commands:[{command:"agefreighterB09.open",title:"B09: Open trial"},{command:"agefreighterB09.artifact",title:"B09: Prepare pinned negative artifact"}],configuration:{properties:{"agefreighter.allowDevelopmentRunnerArtifacts":{type:"boolean",default:false,scope:"machine"}}}}});
  await mkdir(join(root,"user-data/User"),{recursive:true,mode:0o700});await mkdir(join(root,"extensions"),{mode:0o700});
  await write("user-data/User/settings.json",JSON.stringify({"extensions.autoUpdate":false,"extensions.autoCheckUpdates":false,"agefreighter.allowDevelopmentRunnerArtifacts":true}));
  const args=[`--extensionDevelopmentPath=${root}`,"--user-data-dir",join(root,"user-data"),"--extensions-dir",join(root,"extensions"),"--skip-welcome","--skip-release-notes","--new-window"];
  const sourceHashes:Record<string,string>={};for(const name of Object.keys(result.metafile!.inputs)){try{sourceHashes[name]=sha(await readFile(name));}catch{assert.ok(name.startsWith("b09:"));}}
  await retainLostResponseFile(root,"preparation.json",{bundleSHA256,scopeSHA256,initialRecordSHA256,continuation,sourceHashes,args,cloudExecuted:false,nativeQualified:false,sourceDispatchPermitted:false,approvalRequiredBeforeLaunch:true});return args;
}
