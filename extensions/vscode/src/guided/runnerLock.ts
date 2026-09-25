import { execFile } from "node:child_process";
import { readFile, readlink } from "node:fs/promises";
import { promisify } from "node:util";

export interface RunnerLockEnvironment {
  bootIdentity(): Promise<string | undefined>;
  ownerStatus(pid: number): "alive" | "absent" | "uncertain";
  now(): number;
}

/** A boot UUID is kernel evidence, never an estimate from wall clock/uptime.
 * Linux also binds the PID namespace: containers can share a boot UUID while
 * observing different processes. Unsupported/unreadable identities stay locked. */
export const runnerLockEnvironment: RunnerLockEnvironment = {
  async bootIdentity() {
    try {
      if (process.platform === "darwin") {
        const { stdout } = await promisify(execFile)("/usr/sbin/sysctl", ["-n", "kern.bootsessionuuid"], { timeout: 3000, maxBuffer: 256 });
        const uuid = stdout.trim().toLowerCase();
        return /^[a-f0-9-]{36}$/.test(uuid) ? `darwin:${uuid}` : undefined;
      }
      if (process.platform === "linux") {
        const uuid = (await readFile("/proc/sys/kernel/random/boot_id", "utf8")).trim().toLowerCase();
        const namespace = await readlink("/proc/self/ns/pid");
        return /^[a-f0-9-]{36}$/.test(uuid) && /^pid:\[\d+\]$/.test(namespace) ? `linux:${uuid}:${namespace}` : undefined;
      }
    } catch { /* Unknown identity is deliberately unrecoverable. */ }
    return undefined;
  },
  ownerStatus(pid) {
    try { process.kill(pid, 0); return "alive"; }
    catch (error) { return (error as NodeJS.ErrnoException).code === "ESRCH" ? "absent" : "uncertain"; }
  },
  now: Date.now
};

export interface RunnerLockOwner {
  version: 1;
  workflowId: string;
  token: string;
  pid: number;
  bootIdentity: string | null;
  createdAt: string;
}

export interface RunnerLockReview {
  reviewToken: string;
  workflowId: string;
  owner: RunnerLockOwner;
  lockSHA256: string;
  recordSHA256: string;
  device: number;
  inode: number;
  reviewedAt: number;
  expiresAt: number;
}
