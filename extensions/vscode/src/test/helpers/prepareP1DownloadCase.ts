/** Offline test setup only. Not imported by production or registered as a command. */
import {constants} from "node:fs";
import {chmod,link,lstat,mkdir,open,readdir,realpath,unlink} from "node:fs/promises";
import {createHash,randomBytes} from "node:crypto";
import {basename,dirname,isAbsolute,join,parse,resolve} from "node:path";
import {tmpdir} from "node:os";
import {RunnerRecord,object,parseRunnerInput,runnerNames} from "../../core/runner";
import {maxReportBytes,reportManifest} from "../../core/runnerBlob";
import {p1ProfileForConfiguration,p1ProfileSpec} from "../../core/p1Qualification";
import {RunnerStore} from "../../guided/runnerStore";

export type DownloadCase="wrong-sha256"|"wrong-length";
export interface DownloadCaseInput {recordPath:string;manifestPath:string;caseRoot:string;scenario:DownloadCase;acknowledgeNonsecretInputs:true}
const uuid=/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const sha=/^[a-f0-9]{64}$/;
const maxSnapshotBytes=2*1024*1024;
function fail(message:string):never{throw new Error(message);}
const hash=(data:string|Uint8Array)=>createHash("sha256").update(data).digest("hex");
const recordFields=new Set("schemaVersion id phase input artifact prefix deploymentId vmId template previewHash expiresAt updatedAt hourlyComputeUSD guestCommand readinessReceipts readinessRemovals absentStatusCommands absentReadinessCommands guestReady sourceDraft sourceCA cosmosAccess sourceFiles assessment postgresCatalog assessmentHistory reportTransfers rejectedReportExports storageDeployment csvTransfers developmentUpload upgrade upgradeHistory target targetDraft costAuthorizations resize resizeAuthorization migration migrationContinuations resumeInspection targetDiagnostic targetDiagnosticHistory p1Qualification p1QualificationHistory p1Diagnostic migrationHistory targetRestart".split(" "));
function onlyFields(value:Record<string,unknown>,fields:ReadonlySet<string>):void{
  if(Object.keys(value).some(key=>!fields.has(key)))fail("Unknown input fields are not admitted to this fixture.");
}
function knownNonsecretReference(key:string,value:unknown):boolean{
  const date=(v:unknown)=>typeof v==="string"&&/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{3})?Z$/.test(v)&&Number.isFinite(Date.parse(v));
  if(key==="credential"&&value==="default-azure")return true;
  if(key==="resizeAuthorization"){
    const v=object(value);onlyFields(v,new Set(["binding","approvedAt","deadline"]));
    return typeof v.binding==="string"&&sha.test(v.binding)&&date(v.approvedAt)&&date(v.deadline);
  }
  if(key==="costAuthorizations"){
    if(!Array.isArray(value)||value.length>1000)return false;
    return value.every(item=>{
      const v=object(item);onlyFields(v,new Set(["authorizedAt","previous","current"]));
      const cost=(raw:unknown)=>{const c=object(raw);onlyFields(c,new Set(["deadline","budgetUSD","additionalReserveUSD","hourlyUSD"]));return date(c.deadline)&&[c.budgetUSD,c.additionalReserveUSD,c.hourlyUSD].every(n=>typeof n==="number"&&Number.isFinite(n)&&n>=0);};
      return date(v.authorizedAt)&&cost(v.previous)&&cost(v.current);
    });
  }
  if(key==="disablePasswordAuthentication"&&value===true||key==="passwordAuth"&&["Enabled","Disabled"].includes(String(value)))return true;
  if((key==="administratorPassword"&&JSON.stringify(value)==='{"type":"secureString"}')||
    (key==="administratorLoginPassword"&&value==="[parameters('administratorPassword')]"))return true;
  return key==="password"&&!!value&&typeof value==="object"&&!Array.isArray(value)&&
    Object.keys(value).length===1&&(value as Record<string,unknown>).env==="AGEFREIGHTER_SOURCE_PASSWORD";
}
/** Conservative admission, not a proof that arbitrary input can contain no secret. */
function rejectSensitiveInput(value:unknown,depth=0):void{
  if(depth>64)fail("Input nesting is too deep.");
  if(typeof value==="string"){
    if(/-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bBearer\s+\S+|(?:[?&]|\b)(?:sig|access_token|refresh_token|client_secret|AccountKey|SharedAccessSignature)=|(?:postgres(?:ql)?|neo4j(?:\+s)?|https?):\/\/[^\s/]*:[^\s/]*@|\b(?:password|pwd)\s*=|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/i.test(value))fail("Potential secret-bearing input is refused; provide a reviewed nonsecret export.");
    return;
  }
  if(Array.isArray(value)){for(const item of value)rejectSensitiveInput(item,depth+1);return;}
  if(value&&typeof value==="object")for(const [key,item] of Object.entries(value)){
    if(/password|secret|credential|token|connectionstring|privatekey|accountkey|authorization|protectedparameters|^sas$/i.test(key)&&!knownNonsecretReference(key,item))fail("Potential secret-bearing input is refused; provide a reviewed nonsecret export.");
    if(["__proto__","prototype","constructor"].includes(key))fail("Unexpected structured input key.");
    rejectSensitiveInput(item,depth+1);
  }
}
function parseInput(bytes:Buffer):Record<string,unknown>{
  try{
    const text=new TextDecoder("utf-8",{fatal:true}).decode(bytes),value=object(JSON.parse(text));
    // Reject duplicate keys before retaining original raw bytes: JSON.parse
    // otherwise hides earlier values, including an earlier credential field.
    const stack:(Set<string>|null)[]=[];
    const tokens=text.match(/"(?:\\.|[^"\\])*"|[{}\[\]:,]|[^{}\[\]:,\s]+/g)??[];
    for(let i=0;i<tokens.length;i++){
      const token=tokens[i]!;
      if(token==="{")stack.push(new Set());else if(token==="[")stack.push(null);
      else if(token==="}"||token==="]")stack.pop();
      else if(token.startsWith('"')&&tokens[i+1]===":"){
        const keys=stack[stack.length-1],key=JSON.parse(token) as string;
        if(!keys||keys.has(key))fail("Duplicate input keys are refused.");keys.add(key);
      }
      if(stack.length>64)fail("Input nesting is too deep.");
    }
    rejectSensitiveInput(value);return value;
  }catch{fail("Input must be unambiguous, bounded nonsecret UTF-8 JSON objects; no output was prepared.");}
}
function refuseOperatorPath(path:string):void{
  if(/(?:^|\/)(?:runner-v2|keychains|\.azure)(?:\/|$)|\/user\/globalstorage(?:\/|$)|\/library\/application support\/(?:code(?: - insiders)?|vscodium)(?:\/|$)|\/\.config\/(?:code(?: - insiders)?|vscodium)(?:\/|$)|(?:^|\/)(?:state\.vscdb|secretstorage)(?:\.|\/|$)/i.test(path))fail("Operator storage and credential paths are not valid fixture inputs or destinations.");
}
async function unlinkedPath(path:string):Promise<void>{
  if(!isAbsolute(path)||resolve(path)!==path)fail("Use an absolute canonical path without dot segments.");
  let current=parse(path).root;
  for(const segment of path.slice(current.length).split("/").filter(Boolean)){
    current=join(current,segment);const info=await lstat(current);
    if(info.isSymbolicLink())fail("Symbolic links are refused; on macOS use canonical /private/tmp instead of /tmp.");
  }
}
async function inputBytes(path:string,limit:number):Promise<Buffer>{
  refuseOperatorPath(path);await unlinkedPath(path);
  const before=await lstat(path);
  if(!before.isFile()||before.nlink!==1||before.size<1||before.size>limit||before.uid!==process.getuid!())fail("Input must be a bounded, owned regular file without hard links.");
  const file=await open(path,constants.O_RDONLY|constants.O_NOFOLLOW);
  try{
    const stat=await file.stat();
    if(stat.dev!==before.dev||stat.ino!==before.ino||stat.nlink!==1)fail("Input changed during admission.");
    const data=Buffer.alloc(limit+1);let size=0;
    while(size<data.length){const part=await file.read(data,size,data.length-size,null);if(!part.bytesRead)break;size+=part.bytesRead;}
    const after=await file.stat(),current=await lstat(path);
    if(size!==before.size||after.size!==size||after.mtimeMs!==before.mtimeMs||after.ctimeMs!==before.ctimeMs||current.dev!==before.dev||current.ino!==before.ino||current.isSymbolicLink())fail("Input changed during admission.");
    return data.subarray(0,size);
  }finally{await file.close();}
}
function validateRecord(record:Record<string,unknown>,receipt:Record<string,unknown>):RunnerRecord{
  onlyFields(record,recordFields);
  const r=record as unknown as RunnerRecord,q=r.p1Qualification;
  if(r.schemaVersion!==2||!uuid.test(r.id)||r.phase!=="provisioned")fail("A provisioned version-2 workflow snapshot is required.");
  const input=parseRunnerInput(r.input),names=runnerNames(r.id,input);
  if(r.vmId!==names.vmId||r.deploymentId!==names.deploymentId||record.prefix!==undefined&&record.prefix!==names.prefix)fail("Workflow resource identity is inconsistent.");
  if(r.target?.phase!=="provisioned"||r.migration?.phase!=="finished"||r.migration.verification?.outcome!=="pass"||!uuid.test(r.migration.jobId))fail("A provisioned target and finished counts-pass migration are required.");
  const target=r.target;
  if(!/^[a-z][a-z0-9-]{1,62}$/.test(target.input.serverName)||!new RegExp(`^/subscriptions/${input.subscriptionId}/resourceGroups/[A-Za-z0-9_().-]+/providers/Microsoft.DBforPostgreSQL/flexibleServers/${target.input.serverName}$`).test(target.serverId))fail("Target identity is inconsistent.");
  if(!q||!["exported","pass"].includes(q.phase)||!uuid.test(q.operation)||q.jobId!==r.migration.jobId||q.commandId!==`${r.vmId}/runCommands/af-${q.operation}`||!q.exportCommandId?.startsWith(`${r.vmId}/runCommands/af-`)||!uuid.test(q.exportCommandId.slice(`${r.vmId}/runCommands/af-`.length))||q.exportCommandId===q.commandId||!Number.isFinite(Date.parse(q.startedAt)))fail("An already exported qualification with bound job and command identities is required.");
  onlyFields(q as unknown as Record<string,unknown>,new Set("profile operation commandId jobId artifact startedAt phase exportCommandId sha256 bytes replacesFailedOperation".split(" ")));
  const source=object(object(r.sourceDraft?.configuration).source);
  if(source.type!==input.source.type)fail("Reviewed source type changed.");
  const profile=q.profile??"raw-id";p1ProfileSpec(profile);
  if(profile!==p1ProfileForConfiguration(r.sourceDraft?.configuration))fail("Reviewed source profile changed.");
  for(const artifact of [r.artifact,q.artifact]){
    const url=new URL(artifact.url);
    if(!sha.test(artifact.sha256)||!artifact.version||url.protocol!=="https:"||url.username||url.password||url.search||url.hash)fail("A nonsecret sealed artifact identity is required.");
  }
  onlyFields(receipt,new Set(["workflow","operation","jobId","sha256","bytes","exported"]));
  const manifest=reportManifest({operation:String(receipt.operation),sha256:String(receipt.sha256),bytes:receipt.bytes as number});
  if(receipt.exported!==true||receipt.workflow!==r.id||receipt.jobId!==q.jobId||manifest.operation!==q.operation||manifest.sha256!==q.sha256||manifest.bytes!==q.bytes)fail("Independent exported guest receipt does not match the original qualification seal.");
  return r;
}
export async function disposableCaseParent():Promise<string>{
  if(process.platform==="win32"||!process.getuid)fail("This offline fixture utility currently requires POSIX private directories.");
  return realpath(process.platform==="darwin"?"/private/tmp":tmpdir());
}
async function immutableFile(path:string,data:Buffer|string):Promise<void>{
  const file=await open(path,"wx",0o400);
  try{await file.writeFile(data);await file.sync();}finally{await file.close();}
}
export async function prepareP1DownloadCase(options:DownloadCaseInput):Promise<{caseRoot:string;userDataDir:string;extensionsDir:string;recordPath:string;evidencePath:string}>{
  if(options.acknowledgeNonsecretInputs!==true||!["wrong-sha256","wrong-length"].includes(options.scenario))fail("Explicit nonsecret-input acknowledgment and a supported negative case are required.");
  const parent=await disposableCaseParent(),root=options.caseRoot;
  refuseOperatorPath(root);
  if(!isAbsolute(root)||resolve(root)!==root||dirname(root)!==parent||!/^af-b12-[a-f0-9]{12}$/.test(basename(root)))fail(`Use a fresh case root directly under ${parent}, named af-b12- followed by 12 lowercase hex digits. macOS /tmp aliases are not accepted.`);
  await unlinkedPath(parent);
  try{await lstat(root);fail("Case root must not already exist, even if empty; choose a fresh case name.");}catch(error){if((error as NodeJS.ErrnoException).code!=="ENOENT")throw error;}
  if(options.recordPath===options.manifestPath)fail("Snapshot and independent guest receipt must be separate input files.");
  // Finish every input/content check before creating or retaining any output.
  const recordBytes=await inputBytes(options.recordPath,maxSnapshotBytes),manifestBytes=await inputBytes(options.manifestPath,16384);
  const original=validateRecord(parseInput(recordBytes),parseInput(manifestBytes)),candidate=structuredClone(original);
  const q=candidate.p1Qualification!,originalSeal=reportManifest({operation:q.operation,sha256:q.sha256!,bytes:q.bytes!});
  q.phase="exported";
  if(options.scenario==="wrong-sha256")q.sha256=(q.sha256![0]==="0"?"1":"0")+q.sha256!.slice(1);
  else q.bytes=q.bytes===maxReportBytes?q.bytes-1:q.bytes!+1;
  const expectedSeal=reportManifest({operation:q.operation,sha256:q.sha256!,bytes:q.bytes!});
  await mkdir(root,{mode:0o700});
  const rootInfo=await lstat(root);
  const checkRoot=async()=>{
    const stat=await lstat(root);
    if(!stat.isDirectory()||stat.isSymbolicLink()||stat.dev!==rootInfo.dev||stat.ino!==rootInfo.ino||stat.uid!==process.getuid!()||(stat.mode&0o777)!==0o700)fail("Disposable root identity or private permissions changed; partial evidence is preserved.");
  };
  await checkRoot();
  const originals=join(root,"originals"),evidence=join(root,"evidence"),userDataDir=join(root,"user-data"),extensionsDir=join(root,"extensions");
  const storage=join(userDataDir,"User","globalStorage","rioriost.agefreighter","runner-v2"),stage=join(evidence,`stage-${randomBytes(6).toString("hex")}`);
  for(const path of [originals,evidence,userDataDir,extensionsDir,join(userDataDir,"User"),join(userDataDir,"User","globalStorage"),dirname(storage),storage,stage])await mkdir(path,{mode:0o700});
  await immutableFile(join(originals,"record.json"),recordBytes);await immutableFile(join(originals,"guest-export-receipt.json"),manifestBytes);
  await chmod(originals,0o500);
  // RunnerStore.write uses replacement semantics. Write only in our new private
  // staging store, then atomically publish with a no-replace hard link.
  await new RunnerStore(stage).write(candidate);await checkRoot();await unlinkedPath(storage);
  if((await readdir(storage)).length!==0)fail("Case store changed before publication; partial evidence is preserved.");
  const recordPath=join(storage,`${candidate.id}.json`),stagePath=join(stage,`${candidate.id}.json`);
  await link(stagePath,recordPath);await unlink(stagePath);
  const retained=await new RunnerStore(storage).read(candidate.id);
  if(JSON.stringify(retained)!==JSON.stringify(candidate)||(await readdir(storage)).length!==1)fail("Case publication changed; partial evidence is preserved.");
  await checkRoot();
  const evidencePath=join(evidence,"case.json"),summary={version:1,purpose:"B12 disposable fresh-download negative setup",scenario:options.scenario,
    qualification:"not-run",nativeEntrypoint:"agefreighter.continueRunnerExecution",nativeAction:"Qualify / reconcile full P1 digest (development only)",
    originalRecordSHA256:hash(recordBytes),originalReceiptSHA256:hash(manifestBytes),originalSeal,expectedSeal,
    originalQualificationPhase:original.p1Qualification!.phase,profile:q.profile??"raw-id",workflow:original.id,jobId:q.jobId,
    changes:[...(original.p1Qualification!.phase!=="exported"?[{path:"p1Qualification.phase",before:original.p1Qualification!.phase,after:"exported"}]:[]),
      {path:`p1Qualification.${options.scenario==="wrong-sha256"?"sha256":"bytes"}`,before:options.scenario==="wrong-sha256"?originalSeal.sha256:originalSeal.bytes,after:options.scenario==="wrong-sha256"?expectedSeal.sha256:expectedSeal.bytes}],
    disposableRecordSHA256:hash(JSON.stringify(candidate)),caseRoot:root,userDataDir,extensionsDir,recordPath,
    originalBytesReadOnly:true,reportsCopied:0,networkRequests:0,createdAt:new Date().toISOString()};
  await immutableFile(evidencePath,JSON.stringify(summary,null,2)+"\n");
  return {caseRoot:root,userDataDir,extensionsDir,recordPath,evidencePath};
}
function argumentsForCLI(args:string[]):DownloadCaseInput{
  const values=new Map<string,string>();
  for(let i=0;i<args.length;i++){
    const key=args[i]!;
    if(values.has(key)||!["--record","--manifest","--case-root","--scenario","--acknowledge-nonsecret-inputs"].includes(key))fail("Invalid or duplicate fixture argument.");
    if(key==="--acknowledge-nonsecret-inputs")values.set(key,"true");
    else{const value=args[++i];if(!value||value.startsWith("--"))fail("Missing fixture argument.");values.set(key,value);}
  }
  if(values.size!==5)fail("Required: --record ABSOLUTE_JSON --manifest ABSOLUTE_JSON --case-root CANONICAL_TEMP/af-b12-12HEX --scenario wrong-sha256|wrong-length --acknowledge-nonsecret-inputs.");
  return {recordPath:values.get("--record")!,manifestPath:values.get("--manifest")!,caseRoot:values.get("--case-root")!,scenario:values.get("--scenario") as DownloadCase,acknowledgeNonsecretInputs:true};
}
async function main():Promise<void>{console.log(JSON.stringify(await prepareP1DownloadCase(argumentsForCLI(process.argv.slice(2))),null,2));}
if(require.main===module)main().catch(()=>{
  // Avoid exposing filesystem/JSON/URL exception details. Partial output remains
  // in place and cannot be reused; inspect locally before choosing another root.
  console.error("B12 setup refused or incomplete. Inputs must be explicit nonsecret exports; use a fresh canonical temp case root (macOS: /private/tmp/af-b12- followed by 12 lowercase hex digits, not /tmp). Any partial case is preserved. No network or GUI action was performed.");process.exitCode=1;
});
