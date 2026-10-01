import { AzureLocationSummary } from "./azure";
import { object, RunnerInput } from "./runner";
import { RunnerControl } from "./runnerLifecycle";

export interface PlacementCatalog {
  groups: { name: string }[];
  regions: { name: string; displayName: string }[];
}

export function placementCatalog(groups: unknown[], locations: AzureLocationSummary[]): PlacementCatalog {
  const groupNames = new Set(groups.flatMap(raw => {
    const name = object(raw).name;
    return typeof name === "string" && /^[\w().-]{1,90}$/.test(name) && !name.endsWith(".") ? [name] : [];
  }));
  const regions = new Map(locations.filter(region => /^[a-z0-9]{1,64}$/.test(region.name) && region.name !== "global")
    .map(region => [region.name, { name: region.name, displayName: region.displayName }]));
  return {
    groups: [...groupNames].sort((a,b) => a.localeCompare(b)).map(name => ({ name })),
    regions: [...regions.values()].sort((a,b) => a.displayName.localeCompare(b.displayName))
  };
}

export function assertPlacementSelection(input: RunnerInput, catalog: PlacementCatalog): void {
  if (!catalog.groups.some(group => group.name.toLowerCase() === input.resourceGroup.toLowerCase())) {
    throw new Error("Select an existing migration resource group from the current subscription list. Create a new group in Azure first, then refresh the list.");
  }
  if (!catalog.regions.some(region => region.name === input.region)) {
    throw new Error("Select an Azure region from the current subscription list. Refresh the list if needed.");
  }
}

export interface ComputeSubnet {
  id: string; name: string; vnet: string; resourceGroup: string;
  prefixes: string[]; attachedInterfaces: number; containsSource: boolean;
}

/** ARM discovery is not proof of source connectivity. Never select a subnet by name alone. */
export async function discoverComputeSubnets(control: RunnerControl, subscription: string, region: string, sourceId?: string): Promise<ComputeSubnet[]> {
  if (!/^[a-f0-9-]{36}$/i.test(subscription) || !/^[a-z0-9]+$/.test(region)) throw new Error("Select a runner subscription and region first.");
  const sourceSubnets = new Set<string>();
  if (sourceId && /^\/subscriptions\/[a-f0-9-]{36}\/resourceGroups\/[^/]+\/providers\/Microsoft\.Compute\/virtualMachines\/[^/]+$/i.test(sourceId)) {
    const sourceSubscription = sourceId.split("/")[2]!;
    const source = await control.request(sourceSubscription, `${sourceId}?api-version=2024-07-01`);
    if (source.status !== 200) throw new Error("The selected source VM could not be read. Refresh source discovery before choosing a subnet.");
    const interfaces = object(object(object(source.value).properties).networkProfile).networkInterfaces;
    if (!Array.isArray(interfaces)) throw new Error("The selected source VM has no readable network interfaces.");
    for (const raw of interfaces) {
      const id = object(raw).id;
      if (typeof id !== "string" || !id.toLowerCase().startsWith(`/subscriptions/${sourceSubscription}/`.toLowerCase()) ||
          !/\/providers\/Microsoft\.Network\/networkInterfaces\/[^/]+$/i.test(id)) throw new Error("Source NIC identity could not be verified.");
      const nic = await control.request(sourceSubscription, `${id}?api-version=2024-05-01`);
      if (nic.status !== 200) throw new Error("The source VM network could not be read.");
      const configs = object(object(nic.value).properties).ipConfigurations;
      if (!Array.isArray(configs)) throw new Error("The source NIC has no readable subnet configuration.");
      for (const config of configs) {
        const subnet = object(object(object(config).properties).subnet).id;
        if (typeof subnet === "string") sourceSubnets.add(subnet.toLowerCase());
      }
    }
  }
  const networks = await control.list(subscription, `/subscriptions/${subscription}/providers/Microsoft.Network/virtualNetworks?api-version=2024-05-01`);
  const result: ComputeSubnet[] = [];
  for (const raw of networks) {
    const network = object(raw), properties = object(network.properties);
    if (network.location !== region) continue;
    if (typeof network.id !== "string" || !network.id.toLowerCase().startsWith(`/subscriptions/${subscription}/`.toLowerCase()) ||
        !Array.isArray(properties.subnets)) throw new Error("Azure returned incomplete VNet/subnet discovery.");
    for (const item of properties.subnets) {
      const subnet = object(item), p = object(subnet.properties);
      if (typeof subnet.id !== "string" || typeof subnet.name !== "string" ||
          subnet.id.toLowerCase() !== `${network.id}/subnets/${subnet.name}`.toLowerCase()) throw new Error("Azure returned an invalid subnet identity.");
      if (!Array.isArray(p.delegations) || p.delegations.length ||
          /^(GatewaySubnet|AzureBastionSubnet|RouteServerSubnet)$/i.test(subnet.name)) continue;
      if (p.provisioningState && p.provisioningState !== "Succeeded") continue;
      const prefixes = Array.isArray(p.addressPrefixes) ? p.addressPrefixes : [p.addressPrefix];
      if (!prefixes.length || prefixes.some(x => typeof x !== "string")) throw new Error("Subnet address ranges are unavailable.");
      result.push({ id: subnet.id, name: subnet.name, vnet: String(network.name), resourceGroup: network.id.split("/")[4]!,
        prefixes: prefixes.map(String), attachedInterfaces: Array.isArray(p.ipConfigurations) ? p.ipConfigurations.length : 0,
        containsSource: sourceSubnets.has(subnet.id.toLowerCase()) });
    }
  }
  return result.sort((a, b) => Number(a.containsSource) - Number(b.containsSource) ||
    Number(a.attachedInterfaces > 0) - Number(b.attachedInterfaces > 0) || a.id.localeCompare(b.id));
}
