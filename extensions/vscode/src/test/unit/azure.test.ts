import assert from "node:assert/strict";
import test from "node:test";
import { parseAzureResourceID, parseResourceList, parseResourcePage } from "../../core/azure";

test("parses a complete ARM resource ID", () => {
  assert.deepEqual(
    parseAzureResourceID("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/source"),
    { subscriptionId: "sub", resourceGroup: "rg" }
  );
  assert.throws(() => parseAzureResourceID("source.example"), /complete Azure ARM resource ID/);
});

test("reads only bounded placement fields from an ARM resource list", () => {
  assert.deepEqual(parseResourceList({ value: [{
    id: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/source",
    location: "japaneast",
    zones: ["1"],
    properties: { password: "must-not-cross-boundary" }
  }] }), [{
    id: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/source",
    location: "japaneast",
    zones: ["1"]
  }]);
  assert.throws(() => parseResourceList({ resources: [] }), /invalid resource list/);
});

test("retains only a typed continuation link for pagination", () => {
  assert.deepEqual(parseResourcePage({ value: [], nextLink: "https://management.azure.com/next" }), {
    resources: [],
    nextLink: "https://management.azure.com/next"
  });
  assert.throws(() => parseResourcePage({ value: [], nextLink: { href: "unsafe" } }), /invalid continuation link/);
});
