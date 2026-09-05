import { createHash } from "node:crypto";
import { object } from "./runner";

/** Supplied by the controller's retained job, never by a webview message. */
export interface VerificationExpectation {
  jobId: string;
  fingerprint: string;
  cliVersion: string;
  startedAt: string;
  /** Exact source counts recorded before loading. Keys are v.Label / e.TYPE. */
  labels: Record<string, number>;
}

export interface VerificationEvidence {
  exitCode: number;
  reportJSON: string;
  /** Hash received independently from the retained guest artifact manifest. */
  sha256: string;
}

export interface VerificationDecision {
  outcome: "pass" | "fail" | "incomplete";
  summary: string;
  sha256?: string;
}

/** Fail closed: zero exit status, ARM success and a committed load are not verification. */
export function assessCountsVerification(expected: VerificationExpectation, evidence: VerificationEvidence, now = Date.now()): VerificationDecision {
  try {
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(expected.jobId) || !/^[a-f0-9]{64}$/.test(expected.fingerprint)) throw new Error();
    const labels = Object.keys(expected.labels);
    if (!labels.length || labels.length > 255 || labels.some(key => !/^[ve]\..+/.test(key) || !Number.isSafeInteger(expected.labels[key]) || expected.labels[key]! < 0)) throw new Error();
    if (Buffer.byteLength(evidence.reportJSON) > 4 * 1024 * 1024) throw new Error();
    const digest = createHash("sha256").update(evidence.reportJSON).digest("hex");
    if (digest !== evidence.sha256) throw new Error();
    const doc = object(JSON.parse(evidence.reportJSON));
    const job = object(doc.job);
    const generated = Date.parse(String(doc.generatedAt)), started = Date.parse(expected.startedAt);
    if (doc.schemaVersion !== 1 || doc.command !== "verify" || job.id !== expected.jobId || job.configFingerprint !== expected.fingerprint || doc.agefreighterVersion !== expected.cliVersion || !Number.isFinite(generated) || !Number.isFinite(started) || generated < started || generated > now + 60_000) throw new Error();
    for (const name of ["checks", "errors", "warnings", "incompleteChecks", "sections"]) if (!Array.isArray(doc[name])) throw new Error();
    const checks = (doc.checks as unknown[]).map(object);
    const sections = (doc.sections as unknown[]).map(object);
    if (checks.length > 256 || sections.length > 64 || new Set(checks.map(c => c.id)).size !== checks.length) throw new Error();
    const fields = sections.flatMap(section => {
      if (!Array.isArray(section.fields) || section.fields.length > 256) throw new Error();
      return section.fields.map(object);
    });
    if (doc.outcome === "fail" || (doc.errors as unknown[]).length || [...checks, ...fields].some(c => c.status === "fail")) return { outcome: "fail", summary: "Verification found a failed check. Migration is not complete.", sha256: digest };
    if (evidence.exitCode !== 0 || doc.outcome !== "pass" || (doc.incompleteChecks as unknown[]).length || [...checks, ...fields].some(c => c.status !== "pass")) return { outcome: "incomplete", summary: "Verification coverage is incomplete. Migration is not complete.", sha256: digest };
    for (const id of ["job-status", "graph-generation", "generation-ownership"]) if (!checks.some(c => c.id === id && c.status === "pass")) throw new Error();
    const countSections = sections.filter(s => s.title === "Per-label counts");
    if (countSections.length !== 1) throw new Error();
    const counts = (countSections[0]!.fields as unknown[]).map(object);
    if (counts.length !== labels.length + 1 || new Set(counts.map(c => c.name)).size !== counts.length) throw new Error();
    if (!counts.some(c => c.name === "unclassified.rejects" && c.value === "0" && c.status === "pass")) return { outcome: "fail", summary: "Reject-free migration has not been established.", sha256: digest };
    for (const label of labels) {
      const field = counts.find(c => c.name === label);
      if (!field || typeof field.value !== "string") throw new Error();
      const parts = field.value.split(",").map(p => p.split("="));
      if (parts.some(p => p.length !== 2) || new Set(parts.map(p => p[0])).size !== parts.length) throw new Error();
      const counters = Object.fromEntries(parts);
      const rows = String(expected.labels[label]);
      if (["acceptedRows", "committedRows", "livePhysicalRows", "liveIdentityRows"].some(key => counters[key] !== rows) || counters.rejectedRows !== "0") return { outcome: "fail", summary: "Source, committed and live target counts do not agree, or records were rejected.", sha256: digest };
      if (counters.counterCompleteness !== "complete" || counters.storedPhysicalComparison !== "verified" || counters.physicalIdentityEquality !== "verified") throw new Error();
    }
    return { outcome: "pass", summary: "Exact source and target counts agree with no rejects. Full property-digest qualification is a separate check.", sha256: digest };
  } catch {
    // Do not echo raw documents, credentials, source properties or parse errors.
    return { outcome: "incomplete", summary: "Verification evidence is missing, stale, malformed or belongs to a different job. Migration is not complete." };
  }
}
