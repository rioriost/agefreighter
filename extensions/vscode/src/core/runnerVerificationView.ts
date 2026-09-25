import { escapeHTML } from "./report";
import { VerificationDecision } from "./runnerVerification";

/** Display only the controller's decision, never the report's own outcome.
 * Even a counts PASS is not full property-digest qualification. */
export function migrationVerificationView(decision: VerificationDecision, report: string): {title: string; html: string} {
  const outcome = decision.outcome === "pass" ? "pass" : decision.outcome === "fail" ? "fail" : "incomplete";
  const title = outcome === "pass" ? "AGEFreighter counts verified"
    : outcome === "fail" ? "AGEFreighter verification failed" : "AGEFreighter verification incomplete";
  const heading = outcome === "pass" ? "Counts verification: PASS"
    : outcome === "fail" ? "Verification: FAILED — migration is not complete"
      : "Verification: INCOMPLETE — migration is not complete";
  const scope = outcome === "pass"
    ? "Full property-digest qualification remains a separate check."
    : "Do not treat load completion or a report claiming pass as verified migration.";
  return {title, html: `<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>${heading}</h1><p>${escapeHTML(decision.summary)}</p><p>${scope}</p><details><summary>Retained report (untrusted evidence)</summary><pre>${escapeHTML(report)}</pre></details>`};
}
