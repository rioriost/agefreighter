import {randomUUID} from "node:crypto";
import {object,RunnerRecord} from "./runner";
import {RunnerControl} from "./runnerLifecycle";
import {assertIdleHealth} from "./runnerGuest";
import {targetBudget} from "./runnerTarget";
import {targetDSN} from "./runnerExecution";

export interface TargetDiagnostic {commandId:string;operation:string;phase:"submitted"|"unknown"|"finished"|"failed";submittedAt:string;result?:unknown}

/** Explicit operator reconciliation only for an empty target, never a resume or
 * a failed load with retained metadata. All previous evidence remains archived. */
export async function archiveEmptyTargetFailure(control:RunnerControl,r:RunnerRecord):Promise<RunnerRecord>{
  const m=r.migration,d=r.targetDiagnostic,v=object(d?.result),summary=object(v.summary);
  const age=d?Date.now()-Date.parse(d.submittedAt):NaN;
  const checks=Array.isArray(summary.checks)?summary.checks.map(object):[];
  const check=(id:string,status:string,detail:string)=>checks.filter(c=>c.id===id && c.status===status && c.detail===detail).length===1;
  if(!m || m.phase!=="failed" || m.fingerprint || m.reportSHA256 || d?.phase!=="finished" || v.jobId!==m.jobId || v.exitCode!==0 || !Number.isFinite(age) || age<0 || age>900000 || !Array.isArray(summary.errors) || summary.errors.length!==0 ||
    !check("metadata-schema","unavailable","installed=0 supported=21 pending=0; doctor does not migrate") ||
    !check("target-graph","pass","target graph is absent and create mode may create it"))throw new Error("Fresh diagnostic proof of an absent target graph and absent metadata is required; existing or uncertain loads cannot be cleared.");
  assertIdleHealth(r);if(r.guestCommand && ["submitted","unknown"].includes(r.guestCommand.phase))throw new Error("Reconcile guest work first.");
  const next:RunnerRecord={...r,migrationHistory:[...r.migrationHistory??[],{migration:m,diagnostic:d,archivedAt:new Date().toISOString(),reason:"empty-target-preparation-failure"}]};
  delete next.migration;delete next.targetDiagnostic;
  await control.persist(next);return next;
}
// Fixed read-only CLI operation. Never persists a password or arbitrary command
// supplied by a webview; captures bounded, redacted diagnostics on the guest.
export const targetDiagnosticScript=`#!/bin/bash
set -euo pipefail
set +x
umask 077
python3 - <<'AF_DOCTOR'
import os,json,base64,subprocess,pathlib,urllib.parse,hashlib,re
p=json.loads(base64.b64decode(os.environ.pop('AF_DIAGNOSTIC')))
assert all(re.fullmatch('[a-f0-9-]{36}',p[k]) for k in ['workflow','operation','job'])
root=pathlib.Path('/var/lib/agefreighter/workflows')/p['workflow']
assert not (root/'active').exists()
work=root/('diagnostic-'+p['operation']);work.mkdir(mode=0o700)
job=root/p['job']/'job.json'
assert job.is_file() and not job.is_symlink()
env={'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8','AGEFREIGHTER_TARGET_DSN':p['dsn']}
r=subprocess.run(['/usr/local/bin/agefreighter','doctor','--target',str(job),'--format','json'],env=env,capture_output=True,timeout=120)
secrets=[p['dsn'],urllib.parse.unquote(urllib.parse.urlparse(p['dsn']).password or '')]
def redact(b):
    s=b.decode('utf-8',errors='replace')
    for secret in secrets:
        if secret:s=s.replace(secret,'[REDACTED]')
    return s
out=redact(r.stdout);err=redact(r.stderr)
(work/'doctor.json').write_text(out);(work/'doctor.stderr.log').write_text(err)
try:
    doc=json.loads(out);summary={'outcome':doc.get('outcome'),'checks':doc.get('checks'),'errors':doc.get('errors')}
except Exception:summary={'error':err[:1500]}
print(json.dumps({'workflow':p['workflow'],'operation':p['operation'],'jobId':p['job'],'exitCode':r.returncode,'sha256':hashlib.sha256(out.encode()).hexdigest(),'bytes':len(out.encode()),'summary':summary},separators=(',',':'))[:3900])
AF_DOCTOR
`;
export async function diagnoseTarget(control:RunnerControl,r:RunnerRecord,password?:string):Promise<RunnerRecord>{
  if(r.target?.phase!=="provisioned" || !r.migration || !["failed","finished"].includes(r.migration.phase))throw new Error("A terminal retained migration is required for target diagnosis.");
  if(r.targetDiagnostic){
    const d=r.targetDiagnostic;
    if(!d.commandId.startsWith(r.vmId+"/runCommands/af-"))throw new Error("Diagnostic identity changed.");
    const response=await control.request(r.input.subscriptionId,`${d.commandId}?api-version=2024-07-01&$expand=instanceView`),view=object(object(object(response.value).properties).instanceView);
    if(!["Succeeded","Failed","TimedOut","Canceled"].includes(String(view.executionState)))return r;
    let result:unknown;try{result=JSON.parse(String(view.output));}catch{/* Missing/truncated evidence is not success. */}
    const v=object(result),ok=view.executionState==="Succeeded" && view.exitCode===0 && v.workflow===r.id && v.operation===d.operation && v.jobId===r.migration.jobId;
    const next:RunnerRecord={...r,targetDiagnostic:{...d,phase:ok?"finished":"failed",result:ok?result:undefined}};await control.persist(next);return next;
  }
  targetBudget(r.target.input);assertIdleHealth(r);
  if(!password || r.guestCommand && ["submitted","unknown"].includes(r.guestCommand.phase))throw new Error("Credentials and reconciled guest status are required.");
  const response=await control.request(r.input.subscriptionId,`${r.target.serverId}?api-version=2024-08-01`),s=object(response.value),tags=object(s.tags);
  if(response.status!==200 || tags.workflow!==r.id || tags.application!=="agefreighter" || !["migration-target","csv-migration-target"].includes(String(tags.purpose)) || object(s.properties).state!=="Ready")throw new Error("Target ownership/readiness changed.");
  if((await control.list(r.input.subscriptionId,`${r.vmId}/runCommands?api-version=2024-07-01`)).length>=25)throw new Error("Archive completed command evidence before another diagnostic.");
  const operation=randomUUID(),commandId=`${r.vmId}/runCommands/af-${operation}`;
  if((await control.request(r.input.subscriptionId,`${commandId}?api-version=2024-07-01`)).status!==404)throw new Error("Diagnostic command already exists.");
  const next:RunnerRecord={...r,targetDiagnostic:{operation,commandId,phase:"submitted",submittedAt:new Date().toISOString()}};await control.persist(next);
  try{
    const response=await control.request(r.input.subscriptionId,`${commandId}?api-version=2024-07-01`,"PUT",{location:r.input.region,properties:{source:{script:targetDiagnosticScript},protectedParameters:[{name:"AF_DIAGNOSTIC",value:Buffer.from(JSON.stringify({workflow:r.id,operation,job:r.migration.jobId,dsn:targetDSN(r,password)})).toString("base64")}],timeoutInSeconds:180,asyncExecution:false}});
    if(response.status<200 || response.status>=300)throw new Error();
  }catch{next.targetDiagnostic!.phase="unknown";await control.persist(next);}
  return next;
}
