import type { RunnerRecord } from "./runner";

/** An invalidated/new form cannot operate on a retained host-side workflow. */
export function requirePanelWorkflow(current: RunnerRecord | undefined, workflow: unknown): void {
  if (!current || typeof workflow !== "string" || workflow !== current.id) {
    throw new Error("Reconnect to the intended saved workflow before using its controls.");
  }
}
