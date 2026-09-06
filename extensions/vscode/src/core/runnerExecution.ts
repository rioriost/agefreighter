import {createHash,randomUUID} from "node:crypto";
import {object,RunnerRecord} from "./runner";
import {RunnerControl} from "./runnerLifecycle";
import {assertIdleHealth,dispatchGuest,reconcileGuest} from "./runnerGuest";
import {TargetEvidence,sourceTargetEvidence,targetBudget} from "./runnerTarget";
import {assessCountsVerification,VerificationDecision} from "./runnerVerification";

export interface RunnerMigration {
  operation:string; jobId:string; phase:"submitted"|"accepted"|"running"|"finished"|"failed"|"interrupted";
  startedAt:string; bootId:string; artifactSHA256:string; cliVersion:string;
  evidence:TargetEvidence; guestConfigurationSHA256?:string; fingerprint?:string;
  reportSHA256?:string; reportBytes?:number; exitCode?:number; verification?:VerificationDecision;
}
const sha=/^[a-f0-9]{64}$/;
/** ARM may return a location display name instead of its canonical name. */
export function sameAzureLocation(actual:unknown,expected:string):boolean{
  return typeof actual==="string" && actual.length>0 && actual.replace(/\s/g,"").toLowerCase()===expected.replace(/\s/g,"").toLowerCase();
}
export async function migrationPreflight(control:RunnerControl,r:RunnerRecord,report:string):Promise<TargetEvidence>{
  if(r.migration || r.target?.phase!=="provisioned" || r.resize?.phase!=="finished" || r.upgrade && r.upgrade.phase!=="finished")throw new Error("Complete the private target and same-VM resize; an existing migration must never be replayed.");
  targetBudget(r.target.input);assertIdleHealth(r);
  const capability=r.input.source.type==="csv"?"csv-migration-v1":"neo4j-migration-v1";
  if(!r.guestReady?.capabilities?.includes(capability))throw new Error("Upgrade to a reviewed migration-capable Linux artifact and repeat complete source inventory before migration.");
  const e=sourceTargetEvidence(r,report),p=r.target;
  if(e.configurationSHA256!==p.evidence.configurationSHA256 || e.csvManifestSHA256!==p.evidence.csvManifestSHA256 || e.rows!==p.evidence.rows || e.vertices!==p.evidence.vertices || e.edges!==p.evidence.edges || JSON.stringify(e.labels)!==JSON.stringify(p.evidence.labels) || BigInt(e.storageHighBytes)*125n>BigInt(p.input.storageGiB)*1024n**3n*100n)throw new Error("Source mappings, files, counts or capacity changed after target approval.");
  const response=await control.request(r.input.subscriptionId,`${p.serverId}?api-version=2024-08-01`),s=object(response.value),props=object(s.properties),network=object(props.network),tags=object(s.tags);
  if(response.status!==200 || tags.workflow!==r.id || tags.application!=="agefreighter" || !["migration-target","csv-migration-target"].includes(String(tags.purpose)) || !sameAzureLocation(s.location,r.input.region) || props.availabilityZone!==r.input.zone || props.version!=="18" || props.state!=="Ready" || network.publicNetworkAccess!=="Disabled" || network.delegatedSubnetResourceId!==p.subnetId || network.privateDnsZoneArmResourceId!==p.dnsId || object(s.sku).name!==p.input.postgresSKU)throw new Error("Private target identity, placement, SKU or readiness changed.");
  const preload=await control.request(r.input.subscriptionId,`${p.serverId}/configurations/shared_preload_libraries?api-version=2024-08-01`),pc=object(object(preload.value).properties);
  if(preload.status!==200 || pc.isConfigPendingRestart!==false || !String(pc.value).split(",").map(x=>x.trim()).includes("age"))throw new Error("Apply the approved target preload configuration and reconcile its restart before migration.");
  const vm=await control.request(r.input.subscriptionId,`${r.vmId}?api-version=2024-07-01&$expand=instanceView`),v=object(vm.value),vp=object(v.properties);
  if(vm.status!==200 || object(v.tags).workflow!==r.id || object(vp.hardwareProfile).vmSize!==p.input.loaderSize || vp.provisioningState!=="Succeeded" || !Array.isArray(object(vp.instanceView).statuses) || !(object(vp.instanceView).statuses as unknown[]).some(x=>object(x).code==="PowerState/running"))throw new Error("The sized runner is not running and ready.");
  return e;
}

/** One explicit target restart, only to apply the approved AGE preload value.
 * An uncertain POST is never replayed; pending=false is required to reconcile. */
export async function applyTargetPreload(control:RunnerControl,r:RunnerRecord,approved=false):Promise<RunnerRecord>{
  if(r.target?.phase!=="provisioned" || r.migration)throw new Error("Preload preparation is only available before migration.");
  targetBudget(r.target.input);
  const id=r.target.serverId,sub=r.input.subscriptionId;
  const response=await control.request(sub,`${id}?api-version=2024-08-01`),server=object(response.value),tags=object(server.tags);
  if(response.status!==200 || tags.workflow!==r.id || tags.application!=="agefreighter" || !["migration-target","csv-migration-target"].includes(String(tags.purpose)))throw new Error("Target ownership changed.");
  const config=await control.request(sub,`${id}/configurations/shared_preload_libraries?api-version=2024-08-01`),p=object(object(config.value).properties);
  if(config.status!==200 || p.value!=="pg_stat_statements,age" || typeof p.isConfigPendingRestart!=="boolean")throw new Error("Approved target preload configuration changed or is unavailable.");
  if(!p.isConfigPendingRestart && object(server.properties).state==="Ready"){
    const next:RunnerRecord={...r,targetRestart:{phase:"finished",submittedAt:r.targetRestart?.submittedAt??new Date().toISOString()}};await control.persist(next);return next;
  }
  if(r.targetRestart || !approved)return r;
  if(object(server.properties).state!=="Ready")throw new Error("Wait for target provisioning to finish before restart.");
  const next:RunnerRecord={...r,targetRestart:{phase:"submitted",submittedAt:new Date().toISOString()}};await control.persist(next);
  try{const response=await control.request(sub,`${id}/restart?api-version=2024-08-01`,"POST",{});if(response.status<200||response.status>=300)throw new Error();}
  catch{next.targetRestart!.phase="unknown";await control.persist(next);}
  return next;
}
export function targetDSN(r:RunnerRecord,password:string):string{
  if(!r.target || password.length<24)throw new Error("Retained target credentials unavailable.");
  const host=`${r.target.input.serverName}.postgres.database.azure.com`;
  return `postgresql://afadmin:${encodeURIComponent(password)}@${host}:5432/agefreighter?sslmode=verify-full`;
}
/** Exclusive lock, native approval and private credential channel are required. */
export async function startMigration(control:RunnerControl,r:RunnerRecord,report:string,password:string,sourcePassword?:string):Promise<RunnerRecord>{
  if(r.input.source.type==="neo4j" && !sourcePassword)throw new Error("Enter the read-only Neo4j source password for this approved migration.");
  const evidence=await migrationPreflight(control,r,report),operation=randomUUID();
  const migration:RunnerMigration={operation,jobId:operation,phase:"submitted",startedAt:new Date().toISOString(),bootId:r.guestReady!.bootId,artifactSHA256:r.artifact.sha256,cliVersion:r.artifact.version,evidence};
  const action=r.input.source.type==="csv"?"migrate-csv":"migrate-source";
  const secrets:Record<string,string>={AGEFREIGHTER_TARGET_DSN:targetDSN(r,password)};
  if(r.input.source.type==="neo4j"){
    secrets.AGEFREIGHTER_SOURCE_PASSWORD=sourcePassword!;
  }
  return dispatchGuest(control,{...r,migration},{version:1,workflow:r.id,operation,action,configuration:r.sourceDraft!.configuration,secrets});
}
export async function refreshMigration(control:RunnerControl,r:RunnerRecord):Promise<RunnerRecord>{
  const m=r.migration;if(!m)throw new Error("No retained migration.");
  const pending=r.guestCommand && ["submitted","unknown"].includes(r.guestCommand.phase);
  if(!pending)return dispatchGuest(control,r,{version:1,workflow:r.id,operation:m.operation,action:"status"});
  if(r.guestCommand!.operation!==m.operation || !["migrate-csv","migrate-source","status"].includes(r.guestCommand!.action))throw new Error("Reconcile the other pending guest command first.");
  const checked=await reconcileGuest(control,r);if(!checked.result)return checked.record;
  const s=object(checked.result);
  const expectedAction=r.input.source.type==="csv"?"migrate-csv":"migrate-source";
  if(s.jobId!==m.jobId || s.action!==expectedAction || s.bootId!==m.bootId || typeof s.configSha256!=="string" || !sha.test(s.configSha256) || m.guestConfigurationSHA256 && m.guestConfigurationSHA256!==s.configSha256 || !["accepted","running","finished","failed","interrupted"].includes(String(s.phase)))throw new Error("Retained migration identity changed.");
  const next:RunnerRecord={...checked.record,migration:{...m,phase:s.phase as RunnerMigration["phase"],guestConfigurationSHA256:s.configSha256}};
  if(s.reportBytes!==undefined || s.reportSha256!==undefined){
    if(!Number.isSafeInteger(s.reportBytes) || Number(s.reportBytes)<1 || Number(s.reportBytes)>4*1024*1024 || typeof s.reportSha256!=="string" || !sha.test(s.reportSha256) || typeof s.fingerprint!=="string" || !sha.test(s.fingerprint) || m.reportSHA256 && (m.reportSHA256!==s.reportSha256||m.reportBytes!==s.reportBytes))throw new Error("Invalid migration report manifest or fingerprint.");
    Object.assign(next.migration!,{reportBytes:s.reportBytes,reportSHA256:s.reportSha256,fingerprint:s.fingerprint,exitCode:s.exitCode});
  }
  if(s.phase==="finished" && (s.exitCode!==0 || !next.migration!.reportSHA256))throw new Error("Finished migration has no complete verification artifact.");
  await control.persist(next);return next;
}
export function verifyMigrationReport(r:RunnerRecord,text:string):RunnerRecord{
  const m=r.migration;
  if(!m?.reportSHA256 || !m.fingerprint || m.exitCode===undefined || Buffer.byteLength(text)!==m.reportBytes || createHash("sha256").update(text).digest("hex")!==m.reportSHA256)throw new Error("Import the exact retained migration verification artifact first.");
  const verification=assessCountsVerification({jobId:m.jobId,fingerprint:m.fingerprint,cliVersion:m.cliVersion,startedAt:m.startedAt,labels:m.evidence.labels,vertices:m.evidence.vertices,edges:m.evidence.edges},{exitCode:m.exitCode,reportJSON:text,sha256:m.reportSHA256});
  return {...r,migration:{...m,verification}};
}
