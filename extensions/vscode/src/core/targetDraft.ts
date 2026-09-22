import { createHash } from "node:crypto";
import { RunnerRecord } from "./runner";

export function targetDraftBinding(r: RunnerRecord): string {
  return createHash("sha256").update(JSON.stringify([r.id, r.input, r.artifact, r.sourceDraft, r.sourceCA?.sha256,
    r.assessment?.operation, r.assessment?.reportSHA256])).digest("hex");
}
export function retainTargetDraft(r: RunnerRecord, binding: string, input: NonNullable<RunnerRecord["targetDraft"]>["input"], folder?: string): RunnerRecord {
  if (targetDraftBinding(r) !== binding || r.target && r.target.phase !== "previewed") throw new Error("Workflow changed; target inputs were not overwritten.");
  return { ...r, targetDraft: { binding, input: { ...input }, ...(folder ? { folder } : {}) } };
}
