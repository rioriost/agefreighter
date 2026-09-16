import assert from "node:assert/strict";
import test from "node:test";
import { requirePanelWorkflow } from "../../core/runnerPanelBinding";
import { sourceWorkflowDraft } from "../../core/runner";

const id = "11111111-1111-4111-8111-111111111111";
const current = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: {type: "neo4j", location: "on-premises"} });

test("panel actions require the displayed workflow, not only retained host state", () => {
  requirePanelWorkflow(current, id);
  for (const value of [undefined, null, "", "22222222-2222-4222-8222-222222222222", {id}]) {
    assert.throws(() => requirePanelWorkflow(current, value), /Reconnect/);
  }
});

test("a new panel cannot use a previous panel's workflow even with its ID", () => {
  assert.throws(() => requirePanelWorkflow(undefined, id), /Reconnect/);
  assert.equal(current.id, id); // Retained records are not reset or deleted.
});
