import { createHash } from "node:crypto";
import { object, RunnerRecord, validateWhatIf } from "./runner";
import { extractCapacityEvidence, extractInventoryEvidence } from "./guided";
import { csvAssessmentReady } from "./runnerCSV";
import { existingGroupResources, RunnerControl } from "./runnerLifecycle";
import { assertIdleHealth } from "./runnerGuest";

export interface TargetEvidence {
  operation: string; reportSHA256: string; configurationSHA256: string;
  artifactSHA256: string; sourceType?: "csv" | "neo4j" | "postgresql" | "cosmos-nosql"; csvManifestSHA256?: string; rows: string;
  vertices?: string; edges?: string;
  storageHighBytes: string; labels: Record<string, number>;
}
export type CSVTargetEvidence = TargetEvidence;
export interface TargetInput {
  serverName: string; subnetCIDR: string; postgresSKU: string;
  postgresTier: "GeneralPurpose" | "MemoryOptimized"; storageGiB: number;
  loaderSize: string;
  /** Reviewable total reserve including already accrued and non-compute costs. */
  hourlyUSD: number; additionalReserveUSD: number; budgetUSD: number; deadline: string;
}
export interface RunnerTarget {
  phase: "previewed" | "submitted" | "unknown" | "provisioned" | "failed";
  input: TargetInput; evidence: TargetEvidence; template: Record<string,unknown>;
  deploymentId: string; serverId: string; subnetId: string; dnsId: string;
  hash: string; expiresAt: string; generatedAt: string;
  reviewedRunnerInputSHA256?: string;
  /** Present only for a same-subscription VNet in a different existing group. */
  networkDeployment?: { deploymentId: string; resourceGroup: string; vnetId: string; template: Record<string,unknown> };
  configurationRepair?: {phase:"submitted"|"unknown"|"finished";submittedAt:string;originalDeploymentState:"Failed";failureCode:"ServerIsBusy";resourceId:string;previousValue:string;desiredValue:string};
}
const hash = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const sha = /^[a-f0-9]{64}$/;
const csvManifestHash = (record: RunnerRecord) => hash(record.csvTransfers?.map(x => ({file:x.file,sha256:x.sha256,bytes:x.bytes,phase:x.phase})).sort((a,b)=>a.file.localeCompare(b.file)));

/** Only the host's immutable, hash-checked imported artifact is accepted, never
 * report fields posted by a webview or a local CLI standing in for the guest. */
export function csvTargetEvidence(record: RunnerRecord, reportJSON: string): CSVTargetEvidence {
  const assessment=record.assessment;
  if(record.input.source.type!=="csv" || !csvAssessmentReady(record) || !record.sourceDraft?.canAssess || !assessment || assessment.action!=="inventory" || assessment.phase!=="finished" ||
    assessment.configurationSHA256!==hash(record.sourceDraft.configuration) || !record.reportTransfers?.some(x=>x.operation===assessment.operation && x.phase==="imported" && x.sha256===assessment.reportSHA256) ||
    Buffer.byteLength(reportJSON)!==assessment.reportBytes || createHash("sha256").update(reportJSON).digest("hex")!==assessment.reportSHA256) throw new Error("Import a complete, matching guest CSV inventory before planning the target.");
  const doc=object(JSON.parse(reportJSON)), capacity=extractCapacityEvidence(doc), counts=extractInventoryEvidence(doc);
  if(doc.schemaVersion!==1 || doc.command!=="inventory" || doc.agefreighterVersion!==record.artifact.version || !capacity.deployable || !counts.exact || counts.method!=="csv-complete-stream" || counts.totalRows!==capacity.targetRows ||
    !Array.isArray(doc.errors) || doc.errors.length || !Array.isArray(doc.incompleteChecks) || doc.incompleteChecks.length || !Array.isArray(doc.checks) ||
    !["source-counts","source-unchanged","read-only"].every(id=>(doc.checks as unknown[]).some(x=>object(x).id===id && object(x).status==="pass"))) throw new Error("Whole-source CSV evidence is incomplete, changed or from another guest version.");
  const section=(doc.sections as unknown[]).map(object).filter(s=>s.title==="Mapped record counts");
  if((doc.sections as unknown[]).map(object).some(s=>!Array.isArray(s.fields) || s.fields.some(x=>object(x).status!=="pass")))throw new Error("A required source evidence field is not complete.");
  if(section.length!==1 || !Array.isArray(section[0]!.fields))throw new Error("Exact per-label mapped counts are required.");
  const labels:Record<string,number>=Object.create(null);
  for(const raw of section[0]!.fields){const f=object(raw);if(typeof f.name!=="string" || !/^(vertex|edge):[A-Za-z_][A-Za-z0-9_]*$/.test(f.name) || typeof f.value!=="string" || !/^\d+$/.test(f.value) || f.status!=="pass" || !Number.isSafeInteger(Number(f.value)))throw new Error("Invalid mapped label count.");
    const key=f.name.replace(/^vertex:/,"v.").replace(/^edge:/,"e.");if(Object.hasOwn(labels,key))throw new Error("Duplicate mapped label count.");labels[key]=Number(f.value);}
  if(!Object.keys(labels).length || Object.keys(labels).length>255 || Object.values(labels).reduce((a,b)=>a+b,0)!==Number(counts.totalRows))throw new Error("Mapped labels do not cover the whole inventory.");
  if(Object.entries(labels).filter(([key])=>key.startsWith("v.")).reduce((n,[,v])=>n+v,0)!==Number(counts.vertices))throw new Error("Vertex/edge inventory coverage differs from mapped labels.");
  const csv=object(object(record.sourceDraft.configuration.source).csv);
  const expected=new Set([...(csv.vertices as unknown[]).map(x=>`v.${object(x).label}`),...(csv.edges as unknown[]).map(x=>`e.${object(x).label}`)]);
  if(expected.size!==Object.keys(labels).length || Object.keys(labels).some(key=>!expected.has(key)))throw new Error("Inventory labels differ from the reviewed CSV mappings.");
  return {operation:assessment.operation,reportSHA256:assessment.reportSHA256!,configurationSHA256:assessment.configurationSHA256,artifactSHA256:record.artifact.sha256,sourceType:"csv",
    csvManifestSHA256:csvManifestHash(record),rows:counts.totalRows.toString(),vertices:counts.vertices.toString(),edges:counts.edges.toString(),storageHighBytes:capacity.recommendedStorageHigh!.toString(),labels};
}

/** Neo4j's transactional count store provides exact whole-graph totals without
 * scanning customer properties. Until a sampled-width plus exact-count join is
 * retained by the workflow, use a deliberately conservative 16 KiB/record high
 * bound. The user still reviews the resulting target storage and 25% headroom. */
export function neo4jTargetEvidence(record: RunnerRecord, reportJSON: string): TargetEvidence {
  const assessment=record.assessment;
  if(record.input.source.type!=="neo4j" || !record.sourceDraft?.canAssess || !assessment || assessment.action!=="inventory" || assessment.phase!=="finished" ||
    assessment.configurationSHA256!==hash(record.sourceDraft.configuration) || !record.reportTransfers?.some(x=>x.operation===assessment.operation && x.phase==="imported" && x.sha256===assessment.reportSHA256) ||
    Buffer.byteLength(reportJSON)!==assessment.reportBytes || createHash("sha256").update(reportJSON).digest("hex")!==assessment.reportSHA256)throw new Error("Import a complete, matching Neo4j count-store inventory before planning the target.");
  const doc=object(JSON.parse(reportJSON)),counts=extractInventoryEvidence(doc);
  if(doc.schemaVersion!==1 || doc.command!=="inventory" || doc.agefreighterVersion!==record.artifact.version || !counts.exact || counts.method!=="neo4j-transactional-count-store" ||
    !Array.isArray(doc.errors) || doc.errors.length || !Array.isArray(doc.incompleteChecks) || doc.incompleteChecks.length || !Array.isArray(doc.checks) ||
    !(doc.checks as unknown[]).some(x=>object(x).id==="source-counts" && object(x).status==="pass"))throw new Error("Neo4j whole-source count evidence is incomplete, changed or from another guest version.");
  if(!Array.isArray(doc.sections) || (doc.sections as unknown[]).map(object).some(s=>!Array.isArray(s.fields) || s.fields.some(x=>object(x).status!=="pass")))throw new Error("A required Neo4j source evidence field is not complete.");
  const high=counts.totalRows*16384n;
  return {operation:assessment.operation,reportSHA256:assessment.reportSHA256!,configurationSHA256:assessment.configurationSHA256,artifactSHA256:record.artifact.sha256,sourceType:"neo4j",
    rows:counts.totalRows.toString(),vertices:counts.vertices.toString(),edges:counts.edges.toString(),storageHighBytes:high.toString(),labels:Object.create(null)};
}

export function mappedNetworkTargetEvidence(record: RunnerRecord, reportJSON: string): TargetEvidence {
  const type=record.input.source.type,assessment=record.assessment;
  if(!["postgresql","cosmos-nosql"].includes(type) || !record.sourceDraft?.canAssess || !assessment || assessment.action!=="inventory" || assessment.phase!=="finished" ||
    assessment.configurationSHA256!==hash(record.sourceDraft.configuration) || !record.reportTransfers?.some(x=>x.operation===assessment.operation && x.phase==="imported" && x.sha256===assessment.reportSHA256) ||
    Buffer.byteLength(reportJSON)!==assessment.reportBytes || createHash("sha256").update(reportJSON).digest("hex")!==assessment.reportSHA256)throw new Error("Import a complete, matching network-source inventory before planning the target.");
  const doc=object(JSON.parse(reportJSON)),counts=extractInventoryEvidence(doc),capacity=extractCapacityEvidence(doc),expectedMethod=type==="postgresql"?"postgresql-repeatable-read-complete-stream":"cosmos-nosql-complete-stream";
  if(doc.schemaVersion!==1 || doc.command!=="inventory" || doc.agefreighterVersion!==record.artifact.version || !counts.exact || counts.method!==expectedMethod || !capacity.deployable || capacity.recommendedStorageHigh===undefined || counts.totalRows!==capacity.targetRows || !Array.isArray(doc.errors) || doc.errors.length ||
    !Array.isArray(doc.incompleteChecks) || doc.incompleteChecks.length || !Array.isArray(doc.checks) || !["source-counts","read-only"].every(id=>(doc.checks as unknown[]).some(x=>object(x).id===id&&object(x).status==="pass")))throw new Error("Whole-source network inventory evidence is incomplete.");
  if(!Array.isArray(doc.sections) || (doc.sections as unknown[]).map(object).some(s=>!Array.isArray(s.fields)||s.fields.some(x=>object(x).status!=="pass")))throw new Error("A required source evidence field is not complete.");
  const section=(doc.sections as unknown[]).map(object).filter(s=>s.title==="Mapped record counts");
  if(section.length!==1||!Array.isArray(section[0]!.fields))throw new Error("Exact per-label mapped counts are required.");
  const labels:Record<string,number>=Object.create(null);
  for(const raw of section[0]!.fields){const f=object(raw);if(typeof f.name!=="string"||!/^(vertex|edge):[A-Za-z_][A-Za-z0-9_]*$/.test(f.name)||typeof f.value!=="string"||!/^\d+$/.test(f.value)||f.status!=="pass"||!Number.isSafeInteger(Number(f.value)))throw new Error("Invalid mapped label count.");
    const key=f.name.replace(/^vertex:/,"v.").replace(/^edge:/,"e.");if(Object.hasOwn(labels,key))throw new Error("Duplicate mapped label count.");labels[key]=Number(f.value);}
  const keys=Object.keys(labels),vertices=keys.filter(key=>key.startsWith("v."));
  if(!keys.length || keys.length>255 || keys.reduce((n,key)=>n+BigInt(labels[key]!),0n)!==counts.totalRows ||
    vertices.reduce((n,key)=>n+BigInt(labels[key]!),0n)!==counts.vertices)throw new Error("Mapped labels do not cover the whole approved inventory.");
  const source=object(record.sourceDraft.configuration.source),cosmos=object(source.cosmos??{}),gremlin=object(cosmos.gremlin??{}),form=record.sourceDraft.form;
  const discovered=type==="cosmos-nosql" && gremlin.enabled===true;
  if(type==="cosmos-nosql" && (form.cosmosFormat==="gremlin")!==discovered)throw new Error("Gremlin discovery differs from the reviewed source configuration.");
  if(discovered){
    // Gremlin has no manually entered mappings. The hash-bound complete guest
    // inventory is the discovered catalog reviewed by the target approval.
    // Never use a sampled profile or synthesize mappings in persisted state.
    if(source.type!==type || form.mappings.length || cosmos.vertices!==undefined || cosmos.edges!==undefined ||
      typeof gremlin.container!=="string" || !gremlin.container || gremlin.container!==form.container ||
      typeof gremlin.partitionKeyProperty!=="string" || !gremlin.partitionKeyProperty || gremlin.partitionKeyProperty!==form.partitionKey ||
      !Number.isInteger(gremlin.maxLabels) || Number(gremlin.maxLabels)<1 || !vertices.length || vertices.length>Number(gremlin.maxLabels) || keys.length-vertices.length>Number(gremlin.maxLabels))throw new Error("Gremlin catalog differs from the reviewed discovery bounds.");
  }else{
    const expectedLabels=new Set(form.mappings.map(mapping=>`${mapping.kind==="vertex"?"v":"e"}.${mapping.label}`));
    if(expectedLabels.size!==keys.length || keys.some(label=>!expectedLabels.has(label)))throw new Error("Mapped labels do not cover the whole approved inventory.");
  }
  return {operation:assessment.operation,reportSHA256:assessment.reportSHA256!,configurationSHA256:assessment.configurationSHA256,artifactSHA256:record.artifact.sha256,
    sourceType:type as "postgresql"|"cosmos-nosql",rows:counts.totalRows.toString(),vertices:counts.vertices.toString(),edges:counts.edges.toString(),storageHighBytes:capacity.recommendedStorageHigh!.toString(),labels};
}

export function sourceTargetEvidence(record:RunnerRecord,reportJSON:string):TargetEvidence{
  return record.input.source.type==="csv"?csvTargetEvidence(record,reportJSON):record.input.source.type==="neo4j"?neo4jTargetEvidence(record,reportJSON):mappedNetworkTargetEvidence(record,reportJSON);
}

function cidr(value: string): [number,number] {
  const parts=/^(\d+)\.(\d+)\.(\d+)\.(\d+)\/(\d+)$/.exec(value);
  if(!parts)throw new Error("Use a canonical IPv4 subnet CIDR.");
  const n=parts.slice(1).map(Number), bits=n[4]!;
  if(n.slice(0,4).some(x=>x>255) || bits<8 || bits>28)throw new Error("Use an IPv4 subnet between /8 and /28.");
  const start=n[0]!*16777216+n[1]!*65536+n[2]!*256+n[3]!, size=2**(32-bits);
  if(start%size!==0)throw new Error("Subnet CIDR contains host bits.");
  return [start,start+size-1];
}
export function validateTargetSubnet(prefix: string, vnet: unknown): void {
  const [start,end]=cidr(prefix), p=object(object(vnet).properties), space=object(p.addressSpace);
  if(!Array.isArray(space.addressPrefixes) || !space.addressPrefixes.some(x=>{try{const[a,b]=cidr(String(x));return start>=a && end<=b;}catch{return false;}}))throw new Error("Target subnet must fit the existing runner VNet address space.");
  if(!Array.isArray(p.subnets))throw new Error("Complete existing subnet evidence is required.");
  for(const subnet of p.subnets){const sp=object(object(subnet).properties), prefixes=Array.isArray(sp.addressPrefixes)?sp.addressPrefixes:[sp.addressPrefix];
    for(const item of prefixes){const[a,b]=cidr(String(item));if(start<=b && end>=a)throw new Error("Target subnet overlaps an existing subnet; no network mutation is allowed.");}}
}

export function targetBudget(input: TargetInput, now=Date.now()): void {
  const remaining=Date.parse(input.deadline)-now;
  if(!Number.isFinite(remaining) || remaining<=0 || remaining>96*3600000 || !Number.isFinite(input.hourlyUSD) || input.hourlyUSD<=0 || !Number.isFinite(input.additionalReserveUSD) || input.additionalReserveUSD<0 ||
    !Number.isFinite(input.budgetUSD) || input.budgetUSD<=0 || input.hourlyUSD*remaining/3600000+input.additionalReserveUSD>input.budgetUSD)throw new Error("The reviewed remaining-window cost plus accrued/storage/network reserve exceeds the budget or deadline.");
}

/** Records a fresh, explicitly reviewed cost window without touching Azure.
 * The deployed target identity and every non-cost sizing choice are immutable. */
export function renewTargetAuthorization(record: RunnerRecord, input: Pick<TargetInput,"deadline"|"budgetUSD"|"additionalReserveUSD"|"hourlyUSD">, now=Date.now()): RunnerRecord {
  if(record.target?.phase!=="provisioned" || record.migration)throw new Error("Renew authorization only for a provisioned target before migration.");
  const previous=record.target.input;
  const current={...previous,...input};
  targetBudget(current,now);
  const authorization={authorizedAt:new Date(now).toISOString(),previous:{deadline:previous.deadline,budgetUSD:previous.budgetUSD,additionalReserveUSD:previous.additionalReserveUSD,hourlyUSD:previous.hourlyUSD},current:{deadline:current.deadline,budgetUSD:current.budgetUSD,additionalReserveUSD:current.additionalReserveUSD,hourlyUSD:current.hourlyUSD}};
  return {...record,target:{...record.target,input:current},costAuthorizations:[...(record.costAuthorizations??[]),authorization]};
}

export function targetPreview(record: RunnerRecord, input: TargetInput, evidence: TargetEvidence): RunnerTarget {
  if(record.target || record.phase!=="provisioned" || !["csv","neo4j","postgresql","cosmos-nosql"].includes(record.input.source.type) || evidence.sourceType && evidence.sourceType!==record.input.source.type || !sha.test(evidence.reportSHA256))throw new Error("Use an assessed supported-source workflow without an existing target intent.");
  if(!/^[a-z][a-z0-9-]{2,61}[a-z0-9]$/.test(input.serverName) || !/^Standard_[DE]\d+[a-z]*_v[56]$/.test(input.postgresSKU) || !["GeneralPurpose","MemoryOptimized"].includes(input.postgresTier) ||
    !["Standard_D4s_v5","Standard_D8s_v5","Standard_D16s_v5"].includes(input.loaderSize) || ![128,256,512,1024].includes(input.storageGiB))throw new Error("Review a supported private target and x64/SCSI loader size.");
  cidr(input.subnetCIDR);targetBudget(input);
  if(!/^\d+$/.test(evidence.storageHighBytes) || BigInt(evidence.storageHighBytes)*125n>BigInt(input.storageGiB)*1024n**3n*100n)throw new Error("Target storage does not cover the high estimate plus 25% headroom.");
  const base=`/subscriptions/${record.input.subscriptionId}/resourceGroups/${record.input.resourceGroup}`, vnetId=record.input.subnetId.replace(/\/subnets\/[^/]+$/i,"");
  const networkGroup=targetNetworkGroup(record);
  const suffix=record.id.replaceAll("-","").slice(0,20), subnetName=`afpg-${suffix}`, dnsName=`af-${suffix}.postgres.database.azure.com`;
  const serverId=`${base}/providers/Microsoft.DBforPostgreSQL/flexibleServers/${input.serverName}`, subnetId=`${vnetId}/subnets/${subnetName}`, dnsId=`${base}/providers/Microsoft.Network/privateDnsZones/${dnsName}`;
  const tags={application:"agefreighter",workflow:record.id,purpose:"migration-target"};
  const template:Record<string,unknown>={
    $schema:"https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",contentVersion:"1.0.0.0",parameters:{administratorPassword:{type:"secureString"}},resources:[
      {type:"Microsoft.Network/virtualNetworks/subnets",apiVersion:"2024-05-01",name:`${vnetId.split("/").at(-1)}/${subnetName}`,properties:{addressPrefix:input.subnetCIDR,delegations:[{name:"postgres",properties:{serviceName:"Microsoft.DBforPostgreSQL/flexibleServers"}}]}},
      {type:"Microsoft.Network/privateDnsZones",apiVersion:"2024-06-01",name:dnsName,location:"global",tags},
      {type:"Microsoft.Network/privateDnsZones/virtualNetworkLinks",apiVersion:"2024-06-01",name:`${dnsName}/runner`,location:"global",tags,dependsOn:[dnsId],properties:{registrationEnabled:false,virtualNetwork:{id:vnetId}}},
      {type:"Microsoft.DBforPostgreSQL/flexibleServers",apiVersion:"2024-08-01",name:input.serverName,location:record.input.region,tags,dependsOn:[subnetId,`${dnsId}/virtualNetworkLinks/runner`],sku:{name:input.postgresSKU,tier:input.postgresTier},properties:{version:"18",createMode:"Default",administratorLogin:"afadmin",administratorLoginPassword:"[parameters('administratorPassword')]",availabilityZone:record.input.zone,storage:{storageSizeGB:input.storageGiB,autoGrow:"Disabled"},backup:{backupRetentionDays:7,geoRedundantBackup:"Disabled"},highAvailability:{mode:"Disabled"},network:{publicNetworkAccess:"Disabled",delegatedSubnetResourceId:subnetId,privateDnsZoneArmResourceId:dnsId},authConfig:{passwordAuth:"Enabled",activeDirectoryAuth:"Disabled"}}},
      {type:"Microsoft.DBforPostgreSQL/flexibleServers/databases",apiVersion:"2024-08-01",name:`${input.serverName}/agefreighter`,dependsOn:[serverId],properties:{charset:"UTF8",collation:"en_US.utf8"}},
      // Flexible Server rejects concurrent child writes with ServerIsBusy.
      // Order database -> extension allow-list -> preload; external operations
      // can still contend, so a failed deployment must never be auto-replayed.
      ...[["azure.extensions","AGE",`${serverId}/databases/agefreighter`],["shared_preload_libraries","pg_stat_statements,age",`${serverId}/configurations/azure.extensions`]].map(([name,value,after])=>({type:"Microsoft.DBforPostgreSQL/flexibleServers/configurations",apiVersion:"2024-08-01",name:`${input.serverName}/${name}`,dependsOn:[after],properties:{value,source:"user-override"}}))
    ]};
  let networkDeployment:RunnerTarget["networkDeployment"];
  if(networkGroup.toLowerCase()!==record.input.resourceGroup.toLowerCase()){
    const resources=template.resources as Record<string,unknown>[];
    const networkTemplate={$schema:template.$schema,contentVersion:"1.0.0.0",resources:[resources[0]!]};
    const deploymentName=`afpg-net-${suffix}`,deploymentId=`/subscriptions/${record.input.subscriptionId}/resourceGroups/${networkGroup}/providers/Microsoft.Resources/deployments/${deploymentName}`;
    networkDeployment={deploymentId,resourceGroup:networkGroup,vnetId,template:networkTemplate};
    resources[0]={type:"Microsoft.Resources/deployments",apiVersion:"2022-09-01",name:deploymentName,resourceGroup:networkGroup,
      properties:{mode:"Incremental",expressionEvaluationOptions:{scope:"inner"},template:networkTemplate}};
    // The server depends on the scoped deployment, not a child declared in
    // another template. Only ARM executes it; the client never replays it.
    resources[3]!.dependsOn=[deploymentId,`${dnsId}/virtualNetworkLinks/runner`];
  }
  const generatedAt=new Date().toISOString(), expiresAt=new Date(Date.now()+900000).toISOString();
  const plan={phase:"previewed" as const,input,evidence,template,deploymentId:`${base}/providers/Microsoft.Resources/deployments/afpg-${suffix}`,serverId,subnetId,dnsId,generatedAt,expiresAt,reviewedRunnerInputSHA256:hash(record.input),...networkDeployment?{networkDeployment}:{}};
  return {...plan,hash:hash(plan)};
}

export function targetNetworkGroup(record:RunnerRecord):string{
  const match=/^\/subscriptions\/([^/]+)\/resourceGroups\/([A-Za-z0-9_.()-]+)\/providers\/Microsoft\.Network\/virtualNetworks\/([A-Za-z0-9_.-]+)\/subnets\/([A-Za-z0-9_.-]+)$/i.exec(record.input.subnetId);
  if(!match || match[1]!.toLowerCase()!==record.input.subscriptionId.toLowerCase())throw new Error("Target network must be an exact subnet in the runner subscription.");
  return match[2]!;
}

export function assertTargetFresh(record:RunnerRecord): RunnerTarget {
  const p=record.target;if(!p)throw new Error("Review a target preview first.");
  const {hash:retained,...original}=p;
  if(p.reviewedRunnerInputSHA256 && p.reviewedRunnerInputSHA256!==hash(record.input))throw new Error("Reviewed runner input changed; review a new target plan.");
  const networkGroup=targetNetworkGroup(record),vnetId=record.input.subnetId.replace(/\/subnets\/[^/]+$/i,"");
  if(!p.subnetId.toLowerCase().startsWith(`${vnetId}/subnets/`.toLowerCase()) ||
    !p.serverId.toLowerCase().startsWith(`/subscriptions/${record.input.subscriptionId}/resourceGroups/${record.input.resourceGroup}/providers/`.toLowerCase()) ||
    (networkGroup.toLowerCase()!==record.input.resourceGroup.toLowerCase())!==Boolean(p.networkDeployment))throw new Error("Reviewed target network or migration group changed.");
  if(record.phase!=="provisioned" || record.upgrade && record.upgrade.phase!=="finished" || record.guestCommand && ["submitted","unknown"].includes(record.guestCommand.phase) || record.assessment?.phase!=="finished")throw new Error("Reconcile active or uncertain guest operations before target deployment.");
  if(p.phase!=="previewed" || hash(original)!==retained || Date.now()>=Date.parse(p.expiresAt) || !Number.isFinite(Date.parse(p.expiresAt)) || record.artifact.sha256!==p.evidence.artifactSHA256 ||
    !record.sourceDraft || hash(record.sourceDraft.configuration)!==p.evidence.configurationSHA256 ||
    (record.input.source.type==="csv" && (csvManifestHash(record)!==p.evidence.csvManifestSHA256 || !csvAssessmentReady(record))) ||
    (p.evidence.sourceType!==undefined && p.evidence.sourceType!==record.input.source.type) ||
    record.assessment?.operation!==p.evidence.operation || record.assessment?.reportSHA256!==p.evidence.reportSHA256)throw new Error("Target preview or source/artifact evidence changed; no deployment is allowed.");
  targetBudget(p.input);return p;
}
export function targetResourceIds(plan:RunnerTarget):string[]{
  return [plan.subnetId,plan.dnsId,`${plan.dnsId}/virtualNetworkLinks/runner`,plan.serverId,`${plan.serverId}/databases/agefreighter`,`${plan.serverId}/configurations/azure.extensions`,`${plan.serverId}/configurations/shared_preload_libraries`];
}
export async function whatIfTarget(control:RunnerControl,record:RunnerRecord,password:string):Promise<void>{
  const plan=assertTargetFresh(record), sub=record.input.subscriptionId;
  for(const id of [...targetResourceIds(plan),plan.deploymentId,...plan.networkDeployment?[plan.networkDeployment.deploymentId]:[]]){
    const lower=id.toLowerCase(),api=lower.includes("/microsoft.network/")?(lower.includes("/privatednszones/")?"2024-06-01":"2024-05-01"):lower.includes("/deployments/")?"2022-09-01":"2024-08-01";
    if((await control.request(sub,`${id}?api-version=${api}`)).status!==404)throw new Error("A target resource already exists. No resource will be overwritten.");
  }
  const existing=await existingGroupResources(control,record);
  if(plan.networkDeployment){
    const n=plan.networkDeployment;
    const networkExisting=await existingGroupResources(control,{...record,input:{...record.input,resourceGroup:n.resourceGroup}});
    existing.push(...networkExisting);
    // Separate scope preview proves the delegated subnet without passing any
    // database credential to the network deployment.
    await pollTargetWhatIf(control,sub,n.deploymentId,n.template,{},[plan.subnetId],networkExisting);
  }
  await pollTargetWhatIf(control,sub,plan.deploymentId,plan.template,{administratorPassword:{value:password}},targetResourceIds(plan),existing,plan.networkDeployment?.deploymentId);
}
async function pollTargetWhatIf(control:RunnerControl,sub:string,deploymentId:string,template:Record<string,unknown>,parameters:Record<string,unknown>,ids:string[],existing:string[],nestedId?:string):Promise<void>{
  let response=await control.request(sub,`${deploymentId}/whatIf?api-version=2022-09-01`,"POST",{properties:{mode:"Incremental",template,parameters,whatIfSettings:{resultFormat:"ResourceIdOnly"}}});
  for(let i=0;i<30;i++){
    const value=object(response.value);
    if(value.status==="Succeeded"){
      // ARM may include the nested deployment wrapper, but never substitute it
      // for proof of all leaf resources. Reject ignored/unexpanded child plans.
      const changes=object(value.properties).changes;
      const hasWrapper=nestedId && Array.isArray(changes) && changes.some(x=>String(object(x).resourceId).toLowerCase()===nestedId.toLowerCase());
      validateWhatIf(value,hasWrapper?[...ids,nestedId!]:ids,existing);return;
    }
    if(["Failed","Canceled"].includes(String(value.status)) || !response.poll)throw new Error("Target what-if did not produce a complete change review.");
    await control.sleep(2000);const poll=response.poll;response=await control.request(sub,poll);response.poll??=poll;
  }
  throw new Error("Target what-if remains pending. No deployment was submitted.");
}
/** Preflight must recheck service/SKU/quota/network/prices; caller holds the
 * workflow lock and obtained native approval. Secret is never persisted here. */
export async function submitTarget(control:RunnerControl,record:RunnerRecord,password:string,preflight:()=>Promise<void>):Promise<RunnerRecord>{
  assertTargetFresh(record);
  if(password.length<24 || password.length>128 || /[\x00-\x20\x7f]/.test(password))throw new Error("Use a generated strong target password via the private credential channel.");
  await preflight();await whatIfTarget(control,record,password);const p=assertTargetFresh(record);
  const next:RunnerRecord={...record,target:{...p,phase:"submitted"}};await control.persist(next);
  try{const r=await control.request(record.input.subscriptionId,`${p.deploymentId}?api-version=2022-09-01`,"PUT",{properties:{mode:"Incremental",template:p.template,parameters:{administratorPassword:{value:password}}}});if(r.status<200 || r.status>=300)throw new Error();}
  catch{next.target={...next.target!,phase:"unknown"};await control.persist(next);}
  return next;
}
export async function refreshTarget(control:RunnerControl,record:RunnerRecord):Promise<RunnerRecord>{
  const p=record.target;if(!p || p.phase==="previewed")return record;
  if(p.configurationRepair?.phase==="finished")return record;
  if(p.configurationRepair)return repairBusyTargetPreload(control,record);
  const response=await control.request(record.input.subscriptionId,`${p.deploymentId}?api-version=2022-09-01`);
  const state=response.status===404?undefined:object(object(response.value).properties).provisioningState;
  let phase:RunnerTarget["phase"]=state==="Succeeded"?"provisioned":["Failed","Canceled"].includes(String(state))?"failed":["Running","Accepted"].includes(String(state))?"submitted":"unknown";
  if(p.networkDeployment && phase==="provisioned"){
    // Parent status alone is insufficient for the newly supported scope.
    // Failed/missing/inconsistent child evidence is never repaired by replay.
    try{
      const operations=await targetDeploymentOperations(control,record);
      if(operations.some(x=>object(object(x).properties).provisioningState!=="Succeeded"))phase="unknown";
    }catch{phase="unknown";}
  }
  const next:RunnerRecord={...record,target:{...p,phase}};await control.persist(next);return next;
}

/** Flatten only the exact reviewed child after proving its operation identities.
 * Used for cross-group completion and the narrow preload-only repair gate. */
async function targetDeploymentOperations(control:RunnerControl,record:RunnerRecord):Promise<unknown[]>{
  const p=record.target!,sub=record.input.subscriptionId;
  const operations=await control.list(sub,`${p.deploymentId}/operations?api-version=2022-09-01`);
  if(!p.networkDeployment)return operations;
  const n=p.networkDeployment;
  const expected=new Set([...targetResourceIds(p).filter(id=>id!==p.subnetId),n.deploymentId].map(id=>id.toLowerCase()));
  if(operations.length!==expected.size)throw new Error("Incomplete parent deployment operations.");
  const leaves:unknown[]=[];
  for(const entry of operations){
    const op=object(object(entry).properties),id=String(object(op.targetResource).id).toLowerCase();
    if(!expected.delete(id))throw new Error("Unexpected or duplicate parent deployment operation.");
    if(id===n.deploymentId.toLowerCase()){
      if(op.provisioningState!=="Succeeded")throw new Error("Network deployment has not succeeded.");
    }else leaves.push(entry);
  }
  const child=await control.request(sub,`${n.deploymentId}?api-version=2022-09-01`);
  if(child.status!==200 || object(object(child.value).properties).provisioningState!=="Succeeded")throw new Error("Network deployment success evidence is missing.");
  const children=await control.list(sub,`${n.deploymentId}/operations?api-version=2022-09-01`);
  if(children.length!==1)throw new Error("Unexpected network deployment operations.");
  const op=object(object(children[0]).properties);
  if(String(object(op.targetResource).id).toLowerCase()!==p.subnetId.toLowerCase() || op.provisioningState!=="Succeeded")throw new Error("Delegated subnet creation is not proven.");
  return [...leaves,...children];
}

/** Narrow operator recovery, not deployment retry. Only the single failed
 * preload child can be updated once. Retain the original failed deployment,
 * never create resources, retrieve/reset credentials or replay target creation.
 * Reconciliation after any acknowledgement loss is GET-only. */
export async function repairBusyTargetPreload(control:RunnerControl,r:RunnerRecord,approved=false):Promise<RunnerRecord>{
  const p=r.target;
  if(!p || (!p.configurationRepair && p.phase!=="failed") || r.migration || r.resize || r.upgrade && r.upgrade.phase!=="finished" || r.guestCommand && ["submitted","unknown"].includes(r.guestCommand.phase))throw new Error("Repair only a failed pre-migration target after reconciling guest operations.");
  targetBudget(p.input);
  if(!r.sourceDraft?.canAssess || hash(r.sourceDraft.configuration)!==p.evidence.configurationSHA256 || r.artifact.sha256!==p.evidence.artifactSHA256 || r.assessment?.phase!=="finished" || r.assessment.reportSHA256!==p.evidence.reportSHA256)throw new Error("Source or artifact evidence changed before repair.");
  const {hash:retained,configurationRepair,...original}=p;
  if(hash({...original,phase:"previewed"})!==retained)throw new Error("Original reviewed target plan changed; repair is blocked.");
  const sub=r.input.subscriptionId,resourceId=`${p.serverId}/configurations/shared_preload_libraries`,desired="pg_stat_statements,age";
  if(configurationRepair && (configurationRepair.resourceId!==resourceId || configurationRepair.desiredValue!==desired || configurationRepair.failureCode!=="ServerIsBusy" || configurationRepair.originalDeploymentState!=="Failed"))throw new Error("Retained repair identity changed.");
  const deployment=await control.request(sub,`${p.deploymentId}?api-version=2022-09-01`);
  if(deployment.status!==200 || object(object(deployment.value).properties).provisioningState!=="Failed")throw new Error("Original failed deployment evidence changed or is unavailable.");
  const operations=await targetDeploymentOperations(control,r),expected=new Set(targetResourceIds(p).map(x=>x.toLowerCase())),seen=new Set<string>();
  if(operations.length!==expected.size)throw new Error("Review all original target deployment operations before repair.");
  for(const entry of operations){
    const op=object(object(entry).properties),id=String(object(op.targetResource).id).toLowerCase();
    if(!expected.has(id) || seen.has(id))throw new Error("Unexpected or duplicate deployment operation blocks repair.");seen.add(id);
    if(id===resourceId.toLowerCase()){
      if(op.provisioningState!=="Failed" || object(object(op.statusMessage).error).code!=="ServerIsBusy")throw new Error("Only a retained ServerIsBusy preload failure is repairable.");
    }else if(op.provisioningState!=="Succeeded")throw new Error("Other failed or unfinished target resources require separate review.");
  }
  const response=await control.request(sub,`${p.serverId}?api-version=2024-08-01`),server=object(response.value),props=object(server.properties),net=object(props.network),tags=object(server.tags);
  if(response.status!==200 || tags.application!=="agefreighter" || tags.workflow!==r.id || tags.purpose!=="migration-target" || String(server.location).replace(/\s/g,"").toLowerCase()!==r.input.region || props.availabilityZone!==r.input.zone || props.version!=="18" || object(server.sku).name!==p.input.postgresSKU || object(props.storage).storageSizeGB!==p.input.storageGiB || net.publicNetworkAccess!=="Disabled" || String(net.delegatedSubnetResourceId).toLowerCase()!==p.subnetId.toLowerCase() || String(net.privateDnsZoneArmResourceId).toLowerCase()!==p.dnsId.toLowerCase())throw new Error("Target ownership, placement, capacity or private network changed.");
  if(props.state!=="Ready"){
    if(configurationRepair && ["Updating","Starting","Restarting"].includes(String(props.state)))return r;
    throw new Error("Wait for the existing target to become Ready before repair.");
  }
  const db=await control.request(sub,`${p.serverId}/databases/agefreighter?api-version=2024-08-01`),extensions=await control.request(sub,`${p.serverId}/configurations/azure.extensions?api-version=2024-08-01`),preload=await control.request(sub,`${resourceId}?api-version=2024-08-01`),config=object(object(preload.value).properties);
  if(db.status!==200 || extensions.status!==200 || object(object(extensions.value).properties).value!=="AGE" || preload.status!==200 || typeof config.isConfigPendingRestart!=="boolean")throw new Error("Target database or approved AGE settings are missing or changed.");
  if(configurationRepair){
    if(config.value!==desired)return r; // No automatic retry, even after failure.
    const next:RunnerRecord={...r,target:{...p,phase:"provisioned",configurationRepair:{...configurationRepair,phase:"finished"}}};await control.persist(next);return next;
  }
  if(!approved)return r;
  assertIdleHealth(r);targetBudget(p.input);
  if(config.source!=="system-default" || typeof config.defaultValue!=="string" || config.value!==config.defaultValue || config.isConfigPendingRestart!==false)throw new Error("Preload is no longer the unchanged system default; do not overwrite it.");
  const repair:NonNullable<RunnerTarget["configurationRepair"]>={phase:"submitted",submittedAt:new Date().toISOString(),originalDeploymentState:"Failed",failureCode:"ServerIsBusy",resourceId,previousValue:String(config.value),desiredValue:desired};
  let next:RunnerRecord={...r,target:{...p,configurationRepair:repair}};await control.persist(next);
  try{
    const result=await control.request(sub,`${resourceId}?api-version=2024-08-01`,"PUT",{properties:{value:desired,source:"user-override"}});
    if(result.status<200 || result.status>=300)throw new Error();
  }catch{next={...next,target:{...next.target!,configurationRepair:{...repair,phase:"unknown"}}};await control.persist(next);}
  return next;
}
