import assert from "node:assert/strict";
import test from "node:test";
import {
  parseComputeSkus,
  parsePostgresCapabilities,
  parseQuotaUsages,
  parseRetailRates,
  recommendAzure,
  isProposalFresh
} from "../../core/proposal";

const postgresPayload = {
  value: [{
    name: "FlexibleServerCapabilities",
    restricted: "Disabled",
    zoneRedundantHaSupported: "Enabled",
    supportedServerVersions: [{ name: "18", status: "Available" }],
    supportedServerEditions: [{
      name: "MemoryOptimized",
      status: "Available",
      supportedStorageEditions: [{
        name: "ManagedDisk",
        supportedStorageMb: [
          { storageSizeMb: 131072, status: "Available" },
          { storageSizeMb: 1048576, status: "Available" }
        ]
      }],
      supportedServerSkus: [{
        name: "Standard_E8ds_v5",
        vCores: 8,
        supportedMemoryPerVcoreMb: 8192,
        supportedZones: ["1", "2", "3"],
        supportedHaMode: ["SameZone", "ZoneRedundant"],
        status: "Available"
      }]
    }]
  }]
};

const computePayload = {
  value: [{
    resourceType: "virtualMachines",
    name: "Standard_D8ds_v5",
    family: "standardDSv5Family",
    locations: ["japaneast"],
    locationInfo: [{ location: "japaneast", zones: ["1", "2", "3"] }],
    capabilities: [
      { name: "vCPUs", value: "8" },
      { name: "MemoryGB", value: "32" }
    ],
    restrictions: []
  }]
};

test("parses bounded Azure capability and quota fields", () => {
  const postgres = parsePostgresCapabilities(postgresPayload);
  assert.equal(postgres.restricted, false);
  assert.deepEqual(postgres.versions, ["18"]);
  assert.equal(postgres.skus[0]?.memoryPerVCoreMB, 8192);
  assert.deepEqual(postgres.skus[0]?.storageSizesMB, [131072, 1048576]);

  const compute = parseComputeSkus(computePayload, "japaneast");
  assert.deepEqual(compute, [{
    name: "Standard_D8ds_v5",
    family: "standardDSv5Family",
    vCores: 8,
    memoryMB: 32768,
    zones: ["1", "2", "3"],
    restricted: false
  }]);

  assert.deepEqual(parseQuotaUsages({ value: [{
    name: { value: "cores" }, currentValue: 8, limit: 32, ignored: "secret"
  }] }), [{ name: "cores", current: 8, limit: 32 }]);
});

test("excludes a compute SKU restricted in the requested location", () => {
  const restrictedPayload = structuredClone(computePayload);
  const first = restrictedPayload.value[0];
  assert.ok(first);
  first.restrictions = [{
    type: "Location",
    restrictionInfo: { locations: ["japaneast"] }
  }] as never;
  assert.equal(parseComputeSkus(restrictedPayload, "japaneast")[0]?.restricted, true);
});

test("removes only zones affected by a zonal compute restriction", () => {
  const restrictedPayload = structuredClone(computePayload);
  const first = restrictedPayload.value[0];
  assert.ok(first);
  first.restrictions = [{
    type: "Zone",
    restrictionInfo: { locations: ["japaneast"], zones: ["2"] }
  }] as never;
  const parsed = parseComputeSkus(restrictedPayload, "japaneast")[0];
  assert.equal(parsed?.restricted, false);
  assert.deepEqual(parsed?.zones, ["1", "3"]);
});

test("produces a same-zone production-simulation-backed proposal", () => {
  const rates = parseRetailRates({ Items: [
    {
      armSkuName: "Standard_E8ds_v5", serviceName: "Azure Database for PostgreSQL",
      retailPrice: 1.5, effectiveStartDate: "2026-08-01T00:00:00Z",
      currencyCode: "USD", unitOfMeasure: "1 Hour", type: "Consumption"
    },
    {
      armSkuName: "Standard_D8ds_v5", serviceName: "Virtual Machines",
      retailPrice: 0.5, effectiveStartDate: "2026-08-01T00:00:00Z",
      currencyCode: "USD", unitOfMeasure: "1 Hour", type: "Consumption"
    }
  ] });
  const proposal = recommendAzure({
    now: new Date("2026-09-04T00:00:00Z"),
    region: "japaneast",
    sourceZone: "1",
    capacity: {
      method: "exact-counts-scaled-bounded-profile",
      targetRows: 560_000_000n,
      targetRowsLowerBound: false,
      recommendedStorageLow: 400n * 1024n ** 3n,
      recommendedStorageHigh: 600n * 1024n ** 3n,
      deployable: true
    },
    postgres: parsePostgresCapabilities(postgresPayload),
    computeSkus: parseComputeSkus(computePayload, "japaneast"),
    postgresQuota: [{ name: "standardEDSv5Family", current: 0, limit: 64 }],
    computeQuota: [{ name: "standardDSv5Family", current: 0, limit: 64 }],
    rates
  });
  assert.equal(proposal.deployable, true);
  assert.equal(proposal.zone, "1");
  assert.equal(proposal.postgres.sku, "Standard_E8ds_v5");
  assert.equal(proposal.postgres.storageGiB, 1024);
  assert.equal(proposal.loader.sku, "Standard_D8ds_v5");
  assert.equal(proposal.estimatedHourlyUSD, 2);
  assert.deepEqual(proposal.blockers, []);
  assert.equal(isProposalFresh(proposal, new Date("2026-09-04T23:59:59Z")), true);
  assert.equal(isProposalFresh(proposal, new Date("2026-09-05T00:00:00Z")), false);
});

test("proposes a common target zone when the source has no verified zone", () => {
  const proposal = recommendAzure({
    now: new Date("2026-09-04T00:00:00Z"),
    region: "japaneast",
    capacity: {
      method: "exact-counts-scaled-bounded-profile",
      targetRows: 56_000_000n,
      targetRowsLowerBound: false,
      recommendedStorageHigh: 80n * 1024n ** 3n,
      deployable: true
    },
    postgres: parsePostgresCapabilities(postgresPayload),
    computeSkus: parseComputeSkus(computePayload, "japaneast"),
    postgresQuota: [{ name: "standardEDSv5Family", current: 0, limit: 64 }],
    computeQuota: [{ name: "standardDSv5Family", current: 0, limit: 64 }]
  });
  assert.equal(proposal.zone, "1");
  assert.equal(proposal.deployable, true);
  assert.ok(proposal.warnings.some((item) => item.includes("no verified logical zone")));
  assert.ok(proposal.warnings.some((item) => item.includes("retail estimate is incomplete")));
});

test("blocks incomplete sizing evidence and insufficient quotas", () => {
  const proposal = recommendAzure({
    now: new Date("2026-09-04T00:00:00Z"),
    region: "japaneast",
    sourceZone: "1",
    capacity: {
      method: "bounded-prefix",
      targetRows: 56_000_000n,
      targetRowsLowerBound: true,
      deployable: false,
      reason: "Exact inventory is missing."
    },
    postgres: parsePostgresCapabilities(postgresPayload),
    computeSkus: parseComputeSkus(computePayload, "japaneast"),
    postgresQuota: [{ name: "standardEDSv5Family", current: 8, limit: 8 }],
    computeQuota: [{ name: "standardDSv5Family", current: 8, limit: 8 }]
  });
  assert.equal(proposal.deployable, false);
  assert.ok(proposal.blockers.includes("Exact inventory is missing."));
  assert.ok(proposal.blockers.some((item) => item.includes("PostgreSQL quota")));
  assert.ok(proposal.blockers.some((item) => item.includes("Compute quota")));
});

test("blocks unsupported versions, zones, quota, and unqualified scale", () => {
  const postgres = parsePostgresCapabilities(postgresPayload);
  postgres.versions = ["17"];
  const proposal = recommendAzure({
    now: new Date("2026-09-04T00:00:00Z"),
    region: "japaneast",
    sourceZone: "9",
    capacity: {
      method: "complete",
      targetRows: 600_000_000n,
      targetRowsLowerBound: false,
      recommendedStorageHigh: 600n * 1024n ** 3n,
      deployable: true
    },
    postgres,
    computeSkus: parseComputeSkus(computePayload, "japaneast"),
    postgresQuota: [],
    computeQuota: []
  });
  assert.equal(proposal.deployable, false);
  assert.ok(proposal.blockers.some((item) => item.includes("PostgreSQL 18")));
  assert.ok(proposal.warnings.some((item) => item.includes("560-million-row")));
});
