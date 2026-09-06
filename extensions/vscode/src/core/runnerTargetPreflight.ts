import { object, RunnerRecord } from "./runner";
import { RunnerControl, preflightRunner } from "./runnerLifecycle";
import { parsePostgresCapabilities, parseQuotaUsages, RetailRate } from "./proposal";
import { TargetInput, targetBudget, validateTargetSubnet } from "./runnerTarget";

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
export async function preflightTarget(control:RunnerControl,record:RunnerRecord,input:TargetInput):Promise<void>{
  targetBudget(input);
  const sub=record.input.subscriptionId,base=`/subscriptions/${sub}`;
  const vmResponse=await control.request(sub,`${record.vmId}?api-version=2024-07-01&$expand=instanceView`),vm=object(vmResponse.value),p=object(vm.properties),tags=object(vm.tags);
  if(vmResponse.status!==200 || tags.application!=="agefreighter" || tags.workflow!==record.id || tags.purpose!=="discovery-and-migration" || vm.location!==record.input.region || !Array.isArray(vm.zones) || vm.zones.length!==1 || vm.zones[0]!==record.input.zone || p.provisioningState!=="Succeeded")throw new Error("Runner ownership, placement or provisioning evidence changed.");
  const statuses=object(p.instanceView).statuses;
  if(!Array.isArray(statuses) || !statuses.some(x=>object(x).code==="PowerState/running"))throw new Error("Check the running guest before approving target deployment.");
  const ready=record.guestReady,age=ready?Date.now()-Date.parse(ready.checkedAt):NaN;
  if(!ready || !Number.isFinite(age) || age<0 || age>300000 || ready.cliVersion!==record.artifact.version || ready.archiveSha256!==record.artifact.sha256)throw new Error("Fresh matching guest readiness is required for target planning.");
  const nics=object(p.networkProfile).networkInterfaces;
  if(!Array.isArray(nics) || nics.length!==1)throw new Error("Review the runner's single NIC before target deployment.");
  const nicId=object(nics[0]).id;
  if(typeof nicId!=="string" || !nicId.toLowerCase().startsWith(`${base}/resourcegroups/${record.input.resourceGroup}/providers/microsoft.network/networkinterfaces/`.toLowerCase()))throw new Error("Runner NIC is outside this migration group.");
  const nic=await control.request(sub,`${nicId}?api-version=2024-05-01`),configs=object(object(nic.value).properties).ipConfigurations;
  if(nic.status!==200 || !Array.isArray(configs) || configs.length!==1 || object(object(configs[0]).properties).publicIPAddress || object(object(object(configs[0]).properties).subnet).id!==record.input.subnetId)throw new Error("Runner private-network placement changed.");
  await preflightRunner(control,{...record.input,size:input.loaderSize});
  const vnet=await control.request(sub,`${record.input.subnetId.replace(/\/subnets\/[^/]+$/i,"")}?api-version=2024-05-01`);
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
}
