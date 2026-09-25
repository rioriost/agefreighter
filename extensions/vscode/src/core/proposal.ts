import { CapacityEvidence } from "./guided";

const gibibyte = 1024n * 1024n * 1024n;
const mebibyte = 1024n * 1024n;
const qualifiedMaximumRows = 560_000_000n;

export interface PostgresSkuCapability {
  name: string;
  tier: string;
  vCores: number;
  memoryPerVCoreMB: number;
  zones: string[];
  haModes: string[];
  status: string;
  storageSizesMB: number[];
}

export interface PostgresCapabilities {
  restricted: boolean;
  versions: string[];
  zoneRedundantHA: boolean;
  skus: PostgresSkuCapability[];
}

export interface ComputeSkuCapability {
  name: string;
  family: string;
  vCores: number;
  memoryMB: number;
  zones: string[];
  restricted: boolean;
}

export interface QuotaUsage {
  name: string;
  current: number;
  limit: number;
}

export interface RetailRate {
  armSkuName: string;
  serviceName: string;
  hourlyUSD: number;
  effectiveStartDate: string;
}

export interface AzureProposal {
  schemaVersion: 1;
  generatedAt: string;
  expiresAt: string;
  region: string;
  zone?: string;
  postgresVersion: "18";
  age: {
    requestedExtension: "AGE";
    predeploymentEvidence: "published-service-matrix";
    postDeploymentChecks: string[];
  };
  postgres: {
    sku: string;
    tier: string;
    vCores: number;
    storageGiB: number;
    highAvailability: "SameZone";
  };
  loader: {
    sku: string;
    vCores: number;
    memoryGiB: number;
  };
  network: {
    publicAccess: "Disabled";
    privateAccessRequired: true;
  };
  sizingBasis: {
    method: string;
    targetRows: string;
    qualifiedEnvelope: boolean;
    storageHeadroomPercent: 25;
  };
  estimatedHourlyUSD?: number;
  priceEffectiveAt?: string;
  blockers: string[];
  warnings: string[];
  deployable: boolean;
}

export interface ProposalInput {
  now: Date;
  region: string;
  sourceZone?: string;
  capacity: CapacityEvidence;
  postgres: PostgresCapabilities;
  computeSkus: ComputeSkuCapability[];
  postgresQuota: QuotaUsage[];
  computeQuota: QuotaUsage[];
  rates?: RetailRate[];
}

export function parsePostgresCapabilities(value: unknown): PostgresCapabilities {
  const root = record(value);
  const capabilities = array(root?.value).map(record).filter(defined);
  const versions = new Set<string>();
  const skus: PostgresSkuCapability[] = [];
  let restricted = false;
  let zoneRedundantHA = false;
  for (const capability of capabilities) {
    restricted ||= capability.restricted === "Enabled";
    zoneRedundantHA ||= capability.zoneRedundantHaSupported === "Enabled";
    for (const version of array(capability.supportedServerVersions).map(record).filter(defined)) {
      if (typeof version.name === "string" && version.status !== "Disabled") {
        versions.add(version.name);
      }
    }
    for (const edition of array(capability.supportedServerEditions).map(record).filter(defined)) {
      if (typeof edition.name !== "string" || edition.status === "Disabled") {
        continue;
      }
      const storageSizesMB = array(edition.supportedStorageEditions)
        .map(record).filter(defined)
        .flatMap((storage) => array(storage.supportedStorageMb).map(record).filter(defined))
        .filter((storage) => storage.status !== "Disabled")
        .map((storage) => integer(storage.storageSizeMb))
        .filter((size): size is number => size !== undefined && size > 0);
      for (const sku of array(edition.supportedServerSkus).map(record).filter(defined)) {
        const name = text(sku.name);
        const vCores = integer(sku.vCores);
        const memory = integer(sku.supportedMemoryPerVcoreMb);
        if (!name || vCores === undefined || memory === undefined || sku.status === "Disabled") {
          continue;
        }
        skus.push({
          name,
          tier: edition.name,
          vCores,
          memoryPerVCoreMB: memory,
          zones: strings(sku.supportedZones),
          haModes: strings(sku.supportedHaMode),
          status: text(sku.status) ?? "Available",
          storageSizesMB: [...new Set(storageSizesMB)].sort((left, right) => left - right)
        });
      }
    }
  }
  if (capabilities.length === 0) {
    throw new Error("Azure PostgreSQL returned no location capabilities.");
  }
  return { restricted, versions: [...versions].sort(), zoneRedundantHA, skus };
}

export function parseComputeSkus(value: unknown, location: string): ComputeSkuCapability[] {
  const root = record(value);
  if (!root || !Array.isArray(root.value)) {
    throw new Error("Azure Compute returned an invalid SKU list.");
  }
  return root.value.map(record).filter(defined).flatMap((sku): ComputeSkuCapability[] => {
    if (sku.resourceType !== "virtualMachines" || typeof sku.name !== "string" || typeof sku.family !== "string") {
      return [];
    }
    const locations = strings(sku.locations).map(normalize);
    if (!locations.includes(normalize(location))) {
      return [];
    }
    const capabilities = new Map(array(sku.capabilities).map(record).filter(defined)
      .flatMap((item) => typeof item.name === "string" && typeof item.value === "string"
        ? [[item.name.toLocaleLowerCase(), item.value] as const]
        : []));
    const vCores = numericText(capabilities.get("vcpus"));
    const memoryMB = numericText(capabilities.get("memorygb"));
    if (vCores === undefined || memoryMB === undefined) {
      return [];
    }
    const locationInfo = array(sku.locationInfo).map(record).filter(defined)
      .find((item) => normalize(text(item.location) ?? "") === normalize(location));
    let zones = strings(locationInfo?.zones);
    let restricted = false;
    for (const restriction of array(sku.restrictions).map(record).filter(defined)) {
      const info = record(restriction.restrictionInfo);
      const restrictedLocations = strings(info?.locations).map(normalize);
      if (restrictedLocations.length > 0 && !restrictedLocations.includes(normalize(location))) {
        continue;
      }
      if (restriction.type === "Zone") {
        const restrictedZones = new Set(strings(info?.zones));
        zones = zones.filter((zone) => !restrictedZones.has(zone));
      } else {
        restricted = true;
      }
    }
    return [{
      name: sku.name,
      family: sku.family,
      vCores,
      memoryMB: Math.round(memoryMB * 1024),
      zones,
      restricted
    }];
  });
}

export function parseQuotaUsages(value: unknown): QuotaUsage[] {
  const root = record(value);
  if (!root || !Array.isArray(root.value)) {
    throw new Error("Azure returned an invalid quota response.");
  }
  return root.value.map(record).filter(defined).flatMap((usage): QuotaUsage[] => {
    const nameRecord = record(usage.name);
    const name = text(nameRecord?.value);
    const current = number(usage.currentValue);
    const limit = number(usage.limit);
    return name && current !== undefined && limit !== undefined ? [{ name, current, limit }] : [];
  });
}

export function parseRetailRates(value: unknown): RetailRate[] {
  const root = record(value);
  if (!root || !Array.isArray(root.Items)) {
    throw new Error("Azure Retail Prices returned an invalid response.");
  }
  return root.Items.map(record).filter(defined).flatMap((item): RetailRate[] => {
    const armSkuName = text(item.armSkuName);
    const serviceName = text(item.serviceName);
    const hourlyUSD = number(item.retailPrice);
    const effectiveStartDate = text(item.effectiveStartDate);
    const priceLabel = `${text(item.productName) ?? ""} ${text(item.skuName) ?? ""} ${text(item.meterName) ?? ""}`;
    if (!armSkuName || !serviceName || hourlyUSD === undefined || !effectiveStartDate ||
        item.currencyCode !== "USD" || item.unitOfMeasure !== "1 Hour" || item.type !== "Consumption" ||
        // Cloud Services meters share serviceName and armSkuName with ordinary
        // Linux VMs, but are a different product (observed for Bsv2 in Japan East).
        /windows|spot|low priority|cloud services/i.test(priceLabel)) {
      return [];
    }
    return [{ armSkuName, serviceName, hourlyUSD, effectiveStartDate }];
  });
}

export function recommendAzure(input: ProposalInput): AzureProposal {
  const blockers: string[] = [];
  const warnings: string[] = [];
  if (!input.capacity.deployable || input.capacity.targetRows === undefined ||
      input.capacity.recommendedStorageHigh === undefined) {
    blockers.push(input.capacity.reason ?? "Complete capacity evidence is required.");
  }
  if (input.postgres.restricted) {
    blockers.push("Azure Database for PostgreSQL Flexible Server is restricted in this region for the selected subscription.");
  }
  if (!input.postgres.versions.includes("18")) {
    blockers.push("PostgreSQL 18 is not available in this region for the selected subscription.");
  }
  const rows = input.capacity.targetRows ?? 0n;
  const qualifiedEnvelope = rows > 0n && rows <= qualifiedMaximumRows;
  if (!qualifiedEnvelope) {
    warnings.push("This source is outside AGEFreighter's 560-million-row production-simulation sizing envelope; an operator override is required before deployment.");
  }
  const desired = rows <= 1_000_000n
    ? { postgresSKU: "Standard_E4ds_v5", loaderSKU: "Standard_D4s_v5", postgresVCores: 4, loaderVCores: 4, loaderMemoryMB: 16 * 1024 }
    : rows <= qualifiedMaximumRows
      ? { postgresSKU: "Standard_E8ds_v5", loaderSKU: "Standard_D8ds_v5", postgresVCores: 8, loaderVCores: 8, loaderMemoryMB: 32 * 1024 }
      : { postgresSKU: "Standard_E16ds_v5", loaderSKU: "Standard_D16ds_v5", postgresVCores: 16, loaderVCores: 16, loaderMemoryMB: 64 * 1024 };
  const sourceZone = input.sourceZone;
  const postgres = selectPostgresSku(input.postgres.skus, desired.postgresSKU, desired.postgresVCores, sourceZone);
  const loader = selectComputeSku(input.computeSkus, desired.loaderSKU, desired.loaderVCores, desired.loaderMemoryMB, sourceZone);
  if (!postgres) {
    blockers.push("No available Memory Optimized PostgreSQL SKU satisfies the required capacity and source-zone preference.");
  }
  if (!loader) {
    blockers.push("No available loader VM SKU satisfies the required capacity and source-zone preference.");
  }
  let zone = sourceZone ?? commonZone(postgres?.zones ?? [], loader?.zones ?? []);
  if (sourceZone && ((!postgres?.zones.includes(sourceZone)) || (!loader?.zones.includes(sourceZone)))) {
    zone = undefined;
    blockers.push("The source logical zone is not available for both the selected PostgreSQL and loader VM SKUs.");
  }
  if (!zone) {
    blockers.push("No common logical zone is available for the selected PostgreSQL and loader VM SKUs.");
  } else if (!sourceZone) {
    warnings.push(`The source has no verified logical zone; zone ${zone} is a proposed target default that the operator must confirm.`);
  }
  const requiredStorageMB = input.capacity.recommendedStorageHigh === undefined
    ? 0n
    : (input.capacity.recommendedStorageHigh * 125n + (100n * mebibyte - 1n)) / (100n * mebibyte);
  const storageSizes = postgres?.storageSizesMB ?? [];
  const minimumStorageMB = requiredStorageMB > 128n * 1024n ? requiredStorageMB : 128n * 1024n;
  const storageMB = storageSizes.find((size) => BigInt(size) >= minimumStorageMB);
  if (!storageMB) {
    blockers.push("No supported PostgreSQL storage size covers the high estimate plus 25% headroom.");
  }
  if (postgres && !quotaAllows(input.postgresQuota, quotaFamilyFromSku(postgres.name), postgres.vCores)) {
    blockers.push(`PostgreSQL quota is insufficient for ${postgres.name}.`);
  }
  if (loader && !quotaAllows(input.computeQuota, loader.family, loader.vCores)) {
    blockers.push(`Compute quota is insufficient for ${loader.name}.`);
  }
  const matchedRates = input.rates?.filter((rate) =>
    (postgres && same(rate.armSkuName, postgres.name)) || (loader && same(rate.armSkuName, loader.name))) ?? [];
  const postgresRate = postgres ? lowestRate(matchedRates, postgres.name, "Azure Database for PostgreSQL") : undefined;
  const loaderRate = loader ? lowestRate(matchedRates, loader.name, "Virtual Machines") : undefined;
  if (!postgresRate || !loaderRate) {
    warnings.push("The hourly retail estimate is incomplete; storage, networking, discounts, and taxes are excluded in every case.");
  }
  const generatedAt = input.now.toISOString();
  const expiresAt = new Date(input.now.getTime() + 24 * 60 * 60 * 1000).toISOString();
  return {
    schemaVersion: 1,
    generatedAt,
    expiresAt,
    region: input.region,
    zone,
    postgresVersion: "18",
    age: {
      requestedExtension: "AGE",
      predeploymentEvidence: "published-service-matrix",
      postDeploymentChecks: ["azure.extensions contains AGE", "shared_preload_libraries contains AGE", "pg_available_extensions contains age"]
    },
    postgres: {
      sku: postgres?.name ?? "unavailable",
      tier: postgres?.tier ?? "unavailable",
      vCores: postgres?.vCores ?? 0,
      storageGiB: storageMB ? Math.ceil(storageMB / 1024) : 0,
      highAvailability: "SameZone"
    },
    loader: {
      sku: loader?.name ?? "unavailable",
      vCores: loader?.vCores ?? 0,
      memoryGiB: loader ? Math.ceil(loader.memoryMB / 1024) : 0
    },
    network: { publicAccess: "Disabled", privateAccessRequired: true },
    sizingBasis: {
      method: input.capacity.method,
      targetRows: rows.toString(),
      qualifiedEnvelope,
      storageHeadroomPercent: 25
    },
    estimatedHourlyUSD: postgresRate && loaderRate ? postgresRate.hourlyUSD + loaderRate.hourlyUSD : undefined,
    priceEffectiveAt: newestDate([postgresRate, loaderRate].filter(defined).map((rate) => rate.effectiveStartDate)),
    blockers,
    warnings,
    deployable: blockers.length === 0 && qualifiedEnvelope
  };
}

export function isProposalFresh(proposal: AzureProposal, now: Date): boolean {
  const generated = Date.parse(proposal.generatedAt);
  const expires = Date.parse(proposal.expiresAt);
  return Number.isFinite(generated) && Number.isFinite(expires) &&
    generated <= now.getTime() && now.getTime() < expires && expires - generated <= 24 * 60 * 60 * 1000;
}

function selectPostgresSku(
  skus: PostgresSkuCapability[],
  name: string,
  vCores: number,
  zone?: string
): PostgresSkuCapability | undefined {
  return skus.find((sku) => same(sku.name, name) && sku.status !== "Disabled" && sku.tier === "MemoryOptimized" &&
    sku.vCores >= vCores && sku.haModes.includes("SameZone") && (!zone || sku.zones.includes(zone)));
}

function selectComputeSku(
  skus: ComputeSkuCapability[],
  name: string,
  vCores: number,
  memoryMB: number,
  zone?: string
): ComputeSkuCapability | undefined {
  return skus.find((sku) => same(sku.name, name) && !sku.restricted && sku.vCores >= vCores &&
    sku.memoryMB >= memoryMB && (!zone || sku.zones.includes(zone)));
}

function quotaAllows(usages: QuotaUsage[], family: string, required: number): boolean {
  const normalized = normalize(family);
  const usage = usages.find((candidate) => normalize(candidate.name) === normalized) ??
    usages.find((candidate) => normalize(candidate.name).includes(normalized) || normalized.includes(normalize(candidate.name)));
  if (!usage) {
    return false;
  }
  return usage.limit - usage.current >= required;
}

function quotaFamilyFromSku(sku: string): string {
  const match = /^Standard_([A-Za-z]+)\d+([A-Za-z]*)_v(\d+)$/i.exec(sku);
  return match ? `standard${match[1]}${match[2]}v${match[3]}Family` : sku;
}

function commonZone(left: string[], right: string[]): string | undefined {
  return left.filter((zone) => right.includes(zone)).sort((a, b) => a.localeCompare(b))[0];
}

function lowestRate(rates: RetailRate[], sku: string, service: string): RetailRate | undefined {
  return rates.filter((rate) => same(rate.armSkuName, sku) && rate.serviceName === service)
    .sort((left, right) => left.hourlyUSD - right.hourlyUSD)[0];
}

function newestDate(values: string[]): string | undefined {
  return values.sort().at(-1);
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function strings(value: unknown): string[] {
  return array(value).filter((item): item is string => typeof item === "string");
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function integer(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) ? value : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function numericText(value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const result = Number(value);
  return Number.isFinite(result) ? result : undefined;
}

function normalize(value: string): string {
  return value.toLocaleLowerCase().replaceAll(/[^a-z0-9]/g, "");
}

function same(left: string, right: string): boolean {
  return normalize(left) === normalize(right);
}

function defined<T>(value: T | undefined): value is T {
  return value !== undefined;
}

export const proposalConstants = { gibibyte, qualifiedMaximumRows };
