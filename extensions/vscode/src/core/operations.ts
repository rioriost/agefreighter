export const readOperations = [
  "validate",
  "plan",
  "profile",
  "doctor",
  "status",
  "report",
  "optimize"
] as const;

export type ReadOperation = (typeof readOperations)[number];

export const terminalOperations = [
  "load",
  "resume",
  "verify",
  "cleanup"
] as const;

export type TerminalOperation = (typeof terminalOperations)[number];

const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function requiresJobId(operation: ReadOperation | TerminalOperation): boolean {
  return operation === "status" || operation === "report" ||
    operation === "resume" || operation === "verify" || operation === "cleanup";
}

export function validateJobId(jobId: string | undefined): string {
  const value = jobId?.trim();
  if (!value || !uuidPattern.test(value)) {
    throw new Error("A valid durable AGEFreighter job UUID is required.");
  }
  return value;
}

export function buildReadArguments(
  operation: ReadOperation,
  jobPath: string,
  jobId?: string
): string[] {
  switch (operation) {
    case "validate":
      return ["validate", "--format", "json", jobPath];
    case "plan":
      return ["plan", jobPath];
    case "profile":
      return ["profile", "--format", "json", jobPath];
    case "doctor":
      return ["doctor", "--format", "json", "--target", jobPath];
    case "status":
      return ["status", "--target", jobPath, validateJobId(jobId)];
    case "report":
      return ["report", "--format", "json", "--target", jobPath, validateJobId(jobId)];
    case "optimize":
      return ["optimize", "--format", "json", "--target", jobPath];
  }
}

export function buildTerminalArguments(
  operation: TerminalOperation,
  jobPath: string,
  jobId?: string
): string[] {
  switch (operation) {
    case "load":
      return ["load", jobPath];
    case "resume":
      return ["resume", "--job", jobPath, validateJobId(jobId)];
    case "verify":
      return ["verify", "--target", jobPath, validateJobId(jobId)];
    case "cleanup":
      return ["cleanup", "--target", jobPath, validateJobId(jobId)];
  }
}

export function isReadOperation(value: unknown): value is ReadOperation {
  return typeof value === "string" &&
    (readOperations as readonly string[]).includes(value);
}

export function isConnectedReadOperation(operation: ReadOperation): boolean {
  return operation !== "validate" && operation !== "plan";
}
