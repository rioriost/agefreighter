/** Evidence evaluation only; never changes time, workflow data or UI. */
export interface DeploymentExpiryObservation {
  createdAt: string;
  expiresAt: string;
  modalOpenedAt?: string;
  modalReturnedAt?: string;
  nativeChoice?: string;
  deployEntries: number;
  modalEntries: number;
  submitEntries: number;
  storeWrites: number;
  effectAttempts: string[];
  controllerErrors: string[];
  initialSnapshot: Record<string, string>;
  finalSnapshot: Record<string, string>;
}

export function deploymentExpiryFailures(o: DeploymentExpiryObservation): string[] {
  const failures: string[] = [];
  const created = Date.parse(o.createdAt), expires = Date.parse(o.expiresAt);
  const opened = Date.parse(o.modalOpenedAt ?? ""), returned = Date.parse(o.modalReturnedAt ?? "");
  if (![created, expires, opened, returned].every(Number.isFinite) || expires - created !== 900_000 || opened < created || opened >= expires || returned < expires) {
    failures.push("Requires the original fifteen-minute preview, native modal opened before expiry, and real confirmation after expiry.");
  }
  if (o.nativeChoice !== "Create reviewed runner" || o.deployEntries !== 1 || o.modalEntries !== 1) failures.push("Exactly one actual native positive confirmation is required; Cancel or a missing modal is not this case.");
  if (o.submitEntries !== 0 || o.storeWrites !== 0 || o.effectAttempts.length !== 0) failures.push("The expired approval reached an effect boundary.");
  if (o.controllerErrors.length !== 1 || !o.controllerErrors[0]?.includes("approved preview changed or expired")) failures.push("The expected original-review expiry refusal was not observed.");
  if (JSON.stringify(o.initialSnapshot) !== JSON.stringify(o.finalSnapshot)) failures.push("The private workflow store changed after its initial fixture setup.");
  return failures;
}
