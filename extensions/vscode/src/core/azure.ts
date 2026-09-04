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
