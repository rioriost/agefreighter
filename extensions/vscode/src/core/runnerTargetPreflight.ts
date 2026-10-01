import { object, RunnerRecord } from "./runner";
import { RunnerControl, preflightRunner } from "./runnerLifecycle";
import { parsePostgresCapabilities, parseQuotaUsages, RetailRate } from "./proposal";
import { TargetInput, targetBudget, targetNetworkGroup, validateTargetSubnet } from "./runnerTarget";

export const postgresQuotaAPIVersion = "2023-06-01-preview";

/** Exact service/SKU evidence only: never select the cheapest ambiguous meter. */
export function targetComputeRate(rates: RetailRate[], input: TargetInput, now=Date.now()): number {
  const select=(service:string,sku:string)=>{
    const found=rates.filter(r=>r.serviceName===service && r.armSkuName===sku && Number.isFinite(Date.parse(r.effectiveStartDate)) && Date.parse(r.effectiveStartDate)<=now);
    if(found.length!==1 || !Number.isFinite(found[0]!.hourlyUSD) || found[0]!.hourlyUSD<=0)throw new Error("A unique current pay-as-you-go price is required for both target and Linux runner.");
    return found[0]!.hourlyUSD;
  };
  return select("Virtual Machines",input.loaderSize)+select("Azure Database for PostgreSQL",input.postgresSKU);
}

/** Read-only checks, repeated immediately before a separately approved target PUT. */
export async function preflightTarget(control:RunnerControl,record:RunnerRecord,input:TargetInput,
  options:{cancelled?:()=>boolean;progress?:(message:string)=>void}={}):Promise<void>{
  const check=()=>{
    if(options.cancelled?.())throw new Error("Target review cancelled or workspace trust changed. No target deployment was submitted.");
    targetBudget(input);
    const ready=record.guestReady,age=ready?Date.now()-Date.parse(ready.checkedAt):NaN;
    if(!ready || !Number.isFinite(age) || age<0 || age>300000 || ready.cliVersion!==record.artifact.version || ready.archiveSha256!==record.artifact.sha256)throw new Error("Fresh matching guest readiness is required for target planning.");
  };
  check();
  const sub=record.input.subscriptionId,base=`/subscriptions/${sub}`;
  const networkGroup=targetNetworkGroup(record);
  // Both groups must already exist. No inferred group creation or peering.
  for(const group of new Set([record.input.resourceGroup,networkGroup])){
    const result=await control.request(sub,`${base}/resourceGroups/${group}?api-version=2021-04-01`);
    if(result.status!==200)throw new Error("Both migration and network resource groups must exist and be readable.");
  }
  const command=record.guestCommand;
  const finishedReadiness=command?.action==="ready" && command.phase==="finished" &&
    command.id.startsWith(`${record.vmId}/runCommands/af-`) && command.submittedAt===record.guestReady!.checkedAt;
  const until=Date.now()+60000;
  const pending=()=>new Error("Azure VM provisioning is still Updating after Linux readiness. No target deployment was submitted. Wait for Azure to finish, then review the saved target inputs.");
  let p:Record<string,unknown>;
  for(let attempt=0;;attempt++){
    check();
    if(attempt>0 && Date.now()>=until)throw pending();
    const vmResponse=await control.request(sub,`${record.vmId}?api-version=2024-07-01&$expand=instanceView`),vm=object(vmResponse.value),tags=object(vm.tags);
    p=object(vm.properties);
    check();
    if(vmResponse.status!==200 || tags.application!=="agefreighter" || tags.workflow!==record.id || tags.purpose!=="discovery-and-migration" || vm.location!==record.input.region || !Array.isArray(vm.zones) || vm.zones.length!==1 || vm.zones[0]!==record.input.zone)throw new Error("Runner ownership or placement evidence changed.");
    const statuses=object(p.instanceView).statuses;
    if(!Array.isArray(statuses) || !statuses.some(x=>object(x).code==="PowerState/running"))throw new Error("Check the running guest before approving target deployment.");
    if(p.provisioningState==="Succeeded")break;
    // A finished guest receipt can precede the parent VM's ARM completion.
    if(p.provisioningState!=="Updating" || !finishedReadiness)throw new Error("Runner provisioning has not succeeded. Review the VM status and completed Linux readiness before target deployment.");
    if(attempt>=20 || Date.now()>=until)throw pending();
    options.progress?.("Linux readiness passed. Waiting for Azure VM provisioning to finish; status reads only, no operation is resubmitted.");
    await control.sleep(Math.max(0,Math.min(3000,until-Date.now())));
  }
  const nics=object(p.networkProfile).networkInterfaces;
  if(!Array.isArray(nics) || nics.length!==1)throw new Error("Review the runner's single NIC before target deployment.");
  const nicId=object(nics[0]).id;
  if(typeof nicId!=="string" || !nicId.toLowerCase().startsWith(`${base}/resourcegroups/${record.input.resourceGroup}/providers/microsoft.network/networkinterfaces/`.toLowerCase()))throw new Error("Runner NIC is outside this migration group.");
  const nic=await control.request(sub,`${nicId}?api-version=2024-05-01`),configs=object(object(nic.value).properties).ipConfigurations;
  if(nic.status!==200 || !Array.isArray(configs) || configs.length!==1 || object(object(configs[0]).properties).publicIPAddress || object(object(object(configs[0]).properties).subnet).id!==record.input.subnetId)throw new Error("Runner private-network placement changed.");
  await preflightRunner(control,{...record.input,size:input.loaderSize});
  const vnet=await control.request(sub,`${record.input.subnetId.replace(/\/subnets\/[^/]+$/i,"")}?api-version=2024-05-01`);
  if(vnet.status!==200)throw new Error("Read the exact existing runner VNet before target deployment.");
  validateTargetSubnet(input.subnetCIDR,vnet.value);
  const capabilities=parsePostgresCapabilities({value:await control.list(sub,`${base}/providers/Microsoft.DBforPostgreSQL/locations/${record.input.region}/capabilities?api-version=2024-08-01`)});
  const sku=capabilities.skus.find(x=>x.name===input.postgresSKU && x.tier===input.postgresTier && x.zones.includes(record.input.zone));
  if(capabilities.restricted || !capabilities.versions.includes("18") || !sku || sku.status!=="Available" || !sku.storageSizesMB.includes(input.storageGiB*1024))throw new Error("PostgreSQL 18, target SKU, zone or storage is unavailable in this subscription.");
  const family=/^Standard_([DE])\d+([a-z]+)_v([56])$/.exec(input.postgresSKU);
  if(!family)throw new Error("Unknown target quota family.");
  const quota=parseQuotaUsages({value:await control.list(sub,`${base}/providers/Microsoft.DBforPostgreSQL/locations/${record.input.region}/resourceType/flexibleServers/usages?api-version=${postgresQuotaAPIVersion}`)});
  for(const name of ["cores",`standard${family[1]}${family[2]!.toUpperCase()}v${family[3]}Family`]){
    const q=quota.find(x=>x.name.toLowerCase()===name.toLowerCase());
    if(!q || q.limit-q.current<sku.vCores)throw new Error("PostgreSQL regional or family quota is insufficient or unavailable.");
  }
  check();
}
