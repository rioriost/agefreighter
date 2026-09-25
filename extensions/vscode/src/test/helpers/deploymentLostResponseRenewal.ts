/** Explicit, bounded setup-window renewal; never resets an effect claim. */
import assert from "node:assert/strict";
import type { LostResponseScope } from "./deploymentLostResponseStages";
export interface LostResponseSetupRenewal {
  schemaVersion: 1;
  kind: "explicit-ready-storage-setup-renewal";
  previousExpiresAt: string;
  expiresAt: string;
}
export function validateLostResponseRenewal(renewal: LostResponseSetupRenewal, previous: LostResponseScope, current: LostResponseScope, now=Date.now()): void {
  assert.deepEqual(Object.keys(renewal).sort(), ["schemaVersion","kind","previousExpiresAt","expiresAt"].sort());
  assert.equal(renewal.schemaVersion,1); assert.equal(renewal.kind,"explicit-ready-storage-setup-renewal");
  assert.equal(renewal.previousExpiresAt,previous.expiresAt); assert.equal(renewal.expiresAt,current.expiresAt);
  assert.deepEqual({...current,expiresAt:previous.expiresAt},previous,"Only the setup expiry may change during explicit renewal");
  const oldDeadline=Date.parse(previous.expiresAt),deadline=Date.parse(current.expiresAt);
  assert.ok(Number.isFinite(oldDeadline)&&Number.isFinite(deadline)&&oldDeadline<=now&&deadline>oldDeadline&&deadline>now&&deadline-now<=2*3600000,
    "Renewal requires an expired prior scope and a later explicit window of at most two hours");
}
