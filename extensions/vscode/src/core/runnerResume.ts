import {RunnerRecord,object} from "./runner";
import {RunnerControl} from "./runnerLifecycle";
import {dispatchGuest,reconcileGuest} from "./runnerGuest";
import {targetDSN,recoveryIdentity,RunnerMigration} from "./runnerExecution";
import {targetBudget} from "./runnerTarget";
import {randomUUID} from "node:crypto";
import {sourceSecrets} from "./runnerSource";
import {assertCosmosAccessCurrent} from "./runnerCosmosAccess";
import {assertRecoveryRunnerIdentity} from "./runnerResize";

export interface ResumeInspection {
  version:1; workflow:string; operation:string; jobId:string; bootId:string;
  configSha256:string; fingerprint:string; generationId:string; committedRows:string;
  checkpointAt:string; checkedAt:string; outcome:"review-required"; canResume:false; reasons:string[];
}

/** Fresh readiness is read-only, including after a reboot with a retained lease. */
export async function recoveryReadiness(control:RunnerControl,r:RunnerRecord):Promise<RunnerRecord>{
  if(!r.migration || !["failed","interrupted"].includes(r.migration.phase))throw new Error("Refresh the stopped migration before checking recovery readiness.");
  const pending=r.guestCommand?.action==="ready" && ["submitted","unknown"].includes(r.guestCommand.phase);
  let next=pending?r:await dispatchGuest(control,r,{version:1,workflow:r.id,operation:randomUUID(),action:"ready"});
  for(let i=0;i<20;i++){
    next=(await reconcileGuest(control,next)).record;
    if(next.guestCommand?.phase==="finished")return next;
    if(next.guestCommand?.phase==="failed")throw new Error("Recovery readiness failed; retained migration unchanged.");
    await control.sleep(2000);
  }
  return next; // Keep the uncertain read for reconciliation, never replay.
}

export function resumeAdmission(r:RunnerRecord):ResumeInspection{
  const m=r.migration,ready=r.guestReady,h=ready?.health;
  if(!m || !["failed","interrupted"].includes(m.phase) || m.artifactSHA256!==r.artifact.sha256 || m.cliVersion!==r.artifact.version || m.recoveryIdentitySHA256!==recoveryIdentity(r) || r.target?.phase!=="provisioned" || r.resize?.phase!=="finished" || !ready?.capabilities?.includes("explicit-resume-v1"))throw new Error("Recovery requires the original reviewed source, target, runner and pinned build; older unbound jobs cannot be resumed here.");
  targetBudget(r.target.input);
  const i=validateResumeInspection(r,r.resumeInspection);
  if(Date.now()-Date.parse(i.checkpointAt)>900000 || !h || !Number.isFinite(h.storageUsedPercent) || h.storageUsedPercent<0 || h.storageUsedPercent>=80 || h.swapUsedBytes!==0 || h.oomEvents!==0)throw new Error("Recovery checkpoint or guest health is outside the safety gates.");
  return i;
}

/** Caller holds the lock and native approval. Never sends a replacement config. */
export async function resumeMigration(control:RunnerControl,r:RunnerRecord,password:string,sourcePassword?:string,sourceCAPEM?:string):Promise<RunnerRecord>{
  const i=resumeAdmission(r),m=r.migration!;
  if((r.input.source.type==="neo4j"||r.input.source.type==="postgresql")&&!sourcePassword)throw new Error("Enter the source credential for this explicitly approved continuation.");
  await assertCosmosAccessCurrent(control,r);
  await assertRecoveryRunnerIdentity(control,r);
  const response=await control.request(r.input.subscriptionId,`${r.target!.serverId}?api-version=2024-08-01`),s=object(response.value),p=object(s.properties),network=object(p.network),tags=object(s.tags);
  if(response.status!==200||tags.workflow!==r.id||tags.application!=="agefreighter"||p.state!=="Ready"||p.version!=="18"||network.publicNetworkAccess!=="Disabled"||network.delegatedSubnetResourceId!==r.target!.subnetId||network.privateDnsZoneArmResourceId!==r.target!.dnsId)throw new Error("Retained target ownership/network/readiness changed.");
  const secrets:Record<string,string>={AGEFREIGHTER_TARGET_DSN:targetDSN(r,password)};
  if(r.input.source.type==="neo4j"||r.input.source.type==="postgresql")Object.assign(secrets,sourceSecrets(r.input.source.type,r.sourceDraft!.form,sourcePassword,sourceCAPEM));
  resumeAdmission(r);
  const binding={previousOperation:m.operation,jobId:m.jobId,configSha256:i.configSha256,fingerprint:i.fingerprint,generationId:i.generationId,committedRows:i.committedRows};
  const migration:RunnerMigration={operation:randomUUID(),jobId:m.jobId,phase:"submitted",bootId:r.guestReady!.bootId,startedAt:new Date().toISOString(),artifactSHA256:m.artifactSHA256,cliVersion:m.cliVersion,evidence:m.evidence,guestConfigurationSHA256:i.configSha256,fingerprint:i.fingerprint,recoveryIdentitySHA256:m.recoveryIdentitySHA256,resume:binding};
  const next:RunnerRecord={...r,migration,migrationContinuations:[...r.migrationContinuations??[],m]};delete next.resumeInspection;
  return dispatchGuest(control,next,{version:1,workflow:r.id,operation:migration.operation,action:"resume-migration",resume:binding,secrets});
}
const sha=/^[a-f0-9]{64}$/;
/** A successful read is never a resume approval or completed migration. */
export function validateResumeInspection(r:RunnerRecord,raw:unknown):ResumeInspection{
  const v=object(raw),m=r.migration,now=Date.now(),time=Date.parse(String(v.checkedAt)),checkpoint=Date.parse(String(v.checkpointAt));
  if(typeof v.generationId!=="string" || typeof v.committedRows!=="string" || typeof v.configSha256!=="string" || !sha.test(v.configSha256))throw new Error("Recovery identity and counters must be lossless sealed strings.");
  if(!m || v.version!==1 || v.workflow!==r.id || v.operation!==m.operation || v.jobId!==m.jobId || v.bootId!==r.guestReady?.bootId || v.configSha256!==m.guestConfigurationSHA256 || !sha.test(String(v.fingerprint)) || m.fingerprint && m.fingerprint!==v.fingerprint || !/^[1-9][0-9]{0,18}$/.test(String(v.generationId)) || BigInt(String(v.generationId))>9223372036854775807n || !/^(0|[1-9][0-9]{0,18})$/.test(String(v.committedRows)) || BigInt(String(v.committedRows))>BigInt(m.evidence.rows) || !Number.isFinite(time) || time>now || now-time>300000 || !Number.isFinite(checkpoint) || checkpoint>time || v.outcome!=="review-required" || v.canResume!==false || !Array.isArray(v.reasons) || v.reasons.length<1 || v.reasons.length>8 || v.reasons.some(x=>typeof x!=="string"||x.length>200))throw new Error("Resume inspection is missing, stale or does not match the retained migration.");
  return v as unknown as ResumeInspection;
}

/** Call under the workflow lock. Reconciliation is GET-only even after lost PUT. */
export async function inspectResume(control:RunnerControl,r:RunnerRecord,password?:string):Promise<{record:RunnerRecord;inspection?:ResumeInspection}>{
  const m=r.migration;
  if(!m || !["failed","interrupted"].includes(m.phase) || !m.guestConfigurationSHA256 || r.target?.phase!=="provisioned")throw new Error("Refresh a failed/interrupted migration with its sealed configuration first.");
  if(r.guestCommand?.action==="inspect-resume" && ["submitted","unknown"].includes(r.guestCommand.phase)){
    const checked=await reconcileGuest(control,r);
    const inspection=checked.result?validateResumeInspection(checked.record,checked.result):undefined;
    const next={...checked.record};delete next.resumeInspection;
    if(inspection)next.resumeInspection=inspection;
    await control.persist(next);
    return {record:next,inspection};
  }
  if(!r.guestReady?.capabilities?.includes("resume-inspection-v1"))throw new Error("A reviewed runner with resume-inspection-v1 is required; no operation was submitted.");
  if(!password)throw new Error("Retained target credentials required for this read-only check.");
  targetBudget(r.target.input);
  // No readiness refresh/restart here: the caller must first check the current boot.
  const response=await control.request(r.input.subscriptionId,`${r.target.serverId}?api-version=2024-08-01`),s=object(response.value),p=object(s.properties),tags=object(s.tags);
  if(response.status!==200 || tags.workflow!==r.id || tags.application!=="agefreighter" || p.state!=="Ready" || object(p.network).publicNetworkAccess!=="Disabled")throw new Error("Private target ownership/readiness changed.");
  const next={...r};delete next.resumeInspection;
  return {record:await dispatchGuest(control,next,{version:1,workflow:r.id,operation:m.operation,action:"inspect-resume",secrets:{AGEFREIGHTER_TARGET_DSN:targetDSN(r,password)}})};
}
