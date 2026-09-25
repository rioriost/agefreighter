import { AzureLocationSummary } from "./azure";
import { object, RunnerInput } from "./runner";

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
