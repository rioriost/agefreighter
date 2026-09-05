import { object, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";

/** Names are derived from the workflow, never from a SAS or source server. */
export function reportStorageNames(record: RunnerRecord) {
  if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(record.id)) throw new Error("Invalid report workflow.");
  const account = `af${record.id.replaceAll("-", "").slice(0, 22)}`, container = `af-${record.id}`;
  const id = `/subscriptions/${record.input.subscriptionId}/resourceGroups/${record.input.resourceGroup}/providers/Microsoft.Storage/storageAccounts/${account}`;
  return { account, container, id, containerId: `${id}/blobServices/default/containers/${container}`, origin: `https://${account}.blob.core.windows.net` };
}

/** Read-only safety gate, not storage provisioning or permission assignment.
 * An Azure ARM login alone does not imply Blob data-plane access. */
export async function verifyReportStorage(control: RunnerControl, record: RunnerRecord, blob: string): Promise<void> {
  const names = reportStorageNames(record);
  if (!blob.startsWith(`${names.origin}/${names.container}/reports/`)) throw new Error("The report destination is not this workflow's owned storage account.");
  const account = await control.request(record.input.subscriptionId, `${names.id}?api-version=2023-05-01`);
  if (account.status !== 200) throw new Error("Approved private report storage has not been provisioned.");
  const resource = object(account.value), properties = object(resource.properties), tags = object(resource.tags);
  if (String(resource.id).toLowerCase() !== names.id.toLowerCase() || resource.location !== record.input.region ||
    tags.application !== "agefreighter" || tags.workflow !== record.id || tags.purpose !== "artifact-transfer" ||
    properties.provisioningState !== "Succeeded" || properties.supportsHttpsTrafficOnly !== true ||
    properties.allowBlobPublicAccess !== false || properties.allowSharedKeyAccess !== false ||
    !["TLS1_2", "TLS1_3"].includes(String(properties.minimumTlsVersion)) ||
    object(properties.primaryEndpoints).blob !== `${names.origin}/`) throw new Error("Report storage ownership, HTTPS or private-access policy is not verified.");
  const container = await control.request(record.input.subscriptionId, `${names.containerId}?api-version=2023-05-01`);
  if (container.status !== 200) throw new Error("Private report container is unavailable.");
  const target = object(container.value);
  if (String(target.id).toLowerCase() !== names.containerId.toLowerCase() || object(target.properties).publicAccess !== "None") throw new Error("Report container must explicitly disable anonymous access.");
}
