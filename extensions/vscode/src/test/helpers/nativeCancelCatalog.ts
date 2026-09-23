import assert from "node:assert/strict";
import { RunnerRecord } from "../../core/runner";
import { ReportManifest } from "../../core/runnerBlob";
import { buildSourceDraft } from "../../core/runnerSource";
import { cancellationFixture, nativeCancelCases } from "./nativeCancelScenarios";
import { otherCancellationFixture, otherNativeCancelCases } from "./nativeCancelOtherScenarios";
import { sourceCancellationFixture, sourceNativeCancelCases } from "./nativeCancelSourceScenarios";

export const integratedNativeCancelCases = [
  ...nativeCancelCases.map(s => ({ ...s, family: "initial" as const })),
  ...sourceNativeCancelCases.map(s => ({ ...s, family: "source" as const, module: "runnerSourcePanel" as const, exported: "openRunnerSource" as const })),
  ...otherNativeCancelCases.map(s => ({ ...s, family: "other" as const }))
].sort((a, b) => a.id.localeCompare(b.id));
export type IntegratedNativeCancelCase = typeof integratedNativeCancelCases[number];

export function selectNativeCancelCases(selection = ""): IntegratedNativeCancelCase[] {
  if (!selection) return [...integratedNativeCancelCases];
  const ids = selection.split(",");
  assert.ok(ids.length && new Set(ids).size === ids.length, "Fixture case selection must not repeat IDs");
  return ids.map(id => {
    const scenario = integratedNativeCancelCases.find(s => s.id === id);
    assert.ok(scenario, `Unknown or already-credited decision ID: ${id}`); return scenario;
  });
}

export interface IntegratedCancellationFixture {
  record: RunnerRecord; title: string; action: string; responses: Map<string, unknown>; lists: Map<string, unknown[]>;
  reports: { manifest: ReportManifest; text: string }[]; files: { name: string; text: string }[]; openFile?: string;
  inputValues: string[]; selectionIndexes: number[]; prepareMessages: Record<string, unknown>[]; message?: Record<string, unknown>;
  requiresServices: boolean; requiresDevelopmentOptIn: boolean; preparationWrites: number; prerequisiteNote: string;
}

export function integratedCancellationFixture(scenario: IntegratedNativeCancelCase): IntegratedCancellationFixture {
  const common = { lists: new Map<string, unknown[]>(), reports: [], files: [], prepareMessages: [], requiresServices: false,
    requiresDevelopmentOptIn: false, preparationWrites: 0, action: "", prerequisiteNote: "Synthetic workflow/evidence/selector inputs only; no cloud admission claim." };
  if (scenario.family === "initial") return { ...common, ...cancellationFixture(scenario), title: scenario.title, action: scenario.action };
  if (scenario.family === "source") {
    const f = sourceCancellationFixture(scenario);
    return { ...common, ...f, title: scenario.title, preparationWrites: f.baselineAfterPrepare ? 1 : 0 };
  }
  const f = otherCancellationFixture(scenario);
  return { ...common, ...f, preparationWrites: f.preModalObservationWrites };
}

/** Exact pre-consent allowlist; never a generic permission to save new state.
 * A14/A15 permit only the real reviewed SourceDraft, A17 only the same observed
 * failed status. Neither permits timestamp, intent, receipt or report changes. */
export function assertPreparationWrite(scenario: IntegratedNativeCancelCase, fixture: IntegratedCancellationFixture,
  original: RunnerRecord, candidate: RunnerRecord, writesSoFar: number, modalOpened: boolean): void {
  assert.equal(modalOpened, false, "No workflow write is permitted once the native modal opens");
  assert.equal(writesSoFar, 0, "At most one explicitly declared preparation write is permitted");
  assert.equal(fixture.preparationWrites, 1, "This decision permits no preparation persistence");
  if (scenario.id === "A14" || scenario.id === "A15") {
    const message = fixture.prepareMessages[0]; assert.equal(message?.action, "review");
    const sourceDraft = buildSourceDraft(original.input.source, message.form, original.id, original.sourceFiles, original.sourceCA);
    assert.deepEqual(candidate, { ...original, sourceDraft });
  } else {
    assert.equal(scenario.id, "A17"); assert.equal(original.target?.phase, "failed");
    assert.deepEqual(candidate, original, "Preload review permits only unchanged failed-status observation");
  }
}
