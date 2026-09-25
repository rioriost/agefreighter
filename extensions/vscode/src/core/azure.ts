export interface ParsedAzureResourceID {
  subscriptionId: string;
  resourceGroup: string;
}

export interface ResourceSummary {
  id: string;
  location: string;
  zones: string[];
}

export interface ResourcePage {
  resources: ResourceSummary[];
  nextLink?: string;
}

export interface AzureLocationSummary {
  name: string;
  displayName: string;
  physicalLocation?: string;
  latitude?: number;
  longitude?: number;
}

export function parseAzureResourceID(value: string): ParsedAzureResourceID {
  const trimmed = value.trim();
  if (trimmed.length > 2048 || /[\u0000-\u001f\u007f]/u.test(trimmed)) {
    throw new Error("Azure resource ID is too long or contains control characters.");
  }
  const match = /^\/subscriptions\/([^/]+)\/resourceGroups\/([^/]+)\/providers\/[^/]+\/.+$/i.exec(trimmed);
  if (!match?.[1] || !match[2]) {
    throw new Error("Enter a complete Azure ARM resource ID.");
  }
  return { subscriptionId: match[1], resourceGroup: match[2] };
}

export function parseResourcePage(value: unknown): ResourcePage {
  if (value === null || typeof value !== "object" || !Array.isArray((value as { value?: unknown }).value)) {
    throw new Error("Azure Resource Manager returned an invalid resource list.");
  }
  const page = value as { value: unknown[]; nextLink?: unknown };
  const result: ResourceSummary[] = [];
  for (const candidate of page.value) {
    if (candidate === null || typeof candidate !== "object") {
      continue;
    }
    const record = candidate as Record<string, unknown>;
    if (typeof record.id !== "string" || typeof record.location !== "string") {
      continue;
    }
    const zones = Array.isArray(record.zones)
      ? record.zones.filter((zone): zone is string => typeof zone === "string")
      : [];
    result.push({ id: record.id, location: record.location, zones });
  }
  if (page.nextLink !== undefined && typeof page.nextLink !== "string") {
    throw new Error("Azure Resource Manager returned an invalid continuation link.");
  }
  return {
    resources: result,
    nextLink: page.nextLink
  };
}

export function parseResourceList(value: unknown): ResourceSummary[] {
  return parseResourcePage(value).resources;
}

export function parseLocations(value: unknown): AzureLocationSummary[] {
  if (value === null || typeof value !== "object" || !Array.isArray((value as { value?: unknown }).value)) {
    throw new Error("Azure Resource Manager returned an invalid location list.");
  }
  return (value as { value: unknown[] }).value.flatMap((candidate): AzureLocationSummary[] => {
    if (candidate === null || typeof candidate !== "object") {
      return [];
    }
    const item = candidate as Record<string, unknown>;
    if (typeof item.name !== "string" || typeof item.displayName !== "string" ||
        item.type && item.type !== "Region") {
      return [];
    }
    const metadata = item.metadata !== null && typeof item.metadata === "object"
      ? item.metadata as Record<string, unknown>
      : undefined;
    return [{
      name: item.name,
      displayName: item.displayName,
      physicalLocation: typeof metadata?.physicalLocation === "string" ? metadata.physicalLocation : undefined,
      latitude: numericCoordinate(metadata?.latitude),
      longitude: numericCoordinate(metadata?.longitude)
    }];
  }).sort((left, right) => left.displayName.localeCompare(right.displayName));
}

export function recommendRegion(declaredLocation: string, locations: AzureLocationSummary[]): string | undefined {
  const desired = tokens(declaredLocation);
  if (desired.size === 0) {
    return undefined;
  }
  const scored = locations.map((location) => {
    const available = tokens(`${location.displayName} ${location.physicalLocation ?? ""}`);
    let score = 0;
    for (const token of desired) {
      if (available.has(token)) {
        score += token.length;
      }
    }
    return { name: location.name, score };
  }).filter((item) => item.score > 0).sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));
  const first = scored[0];
  const second = scored[1];
  return first && (!second || first.score > second.score) ? first.name : undefined;
}

function numericCoordinate(value: unknown): number | undefined {
  const parsed = typeof value === "string" ? Number(value) : value;
  return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : undefined;
}

function tokens(value: string): Set<string> {
  return new Set(value.toLocaleLowerCase().split(/[^\p{L}\p{N}]+/u).filter((token) => token.length >= 2));
}
