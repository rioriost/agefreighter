import { object, RunnerRecord } from "./runner";
import { sourceTargetEvidence } from "./runnerTarget";

export interface SourceReportSummary {
  title: string; detail: string; canReviewTarget: boolean;
  vertices?: string; edges?: string;
}

export function sourceReportSummary(record: RunnerRecord, text: string): SourceReportSummary {
  const doc = object(JSON.parse(text));
  if (doc.outcome !== "pass") return {
    title: `Source assessment ${typeof doc.outcome === "string" ? doc.outcome : "outcome unavailable"}`,
    detail: "The report hash is verified, but source assessment did not pass. Review the report checks before a fresh approval. For TLS sources, check the selected CA and runner-to-source connectivity. Target review is blocked.",
    canReviewTarget: false
  };
  if (doc.command === "profile") return {
    title: "Sample assessment passed",
    detail: "Next: review source settings and approve a complete inventory. A sample is not whole-source evidence or a migration approval.",
    canReviewTarget: false
  };
  try {
    const evidence = sourceTargetEvidence(record, text);
    return { title: "Complete source inventory passed", detail: "Next: review the private target and migration sizing. No target or migration has been approved.",
      vertices: evidence.vertices, edges: evidence.edges, canReviewTarget: true };
  } catch (error) {
    return { title: "Inventory needs review", detail: error instanceof Error ? error.message : "Whole-source evidence could not be verified. Target review is blocked.", canReviewTarget: false };
  }
}
