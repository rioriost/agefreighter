/** A bounded reconciler, not a retry engine. Each step must read fresh retained
 * state under its own lock. Exceptions stop immediately; sleeps hold no lock. */
export async function boundedWatch<T>(options: {
  step: () => Promise<T>; done: (value: T) => boolean; sleep: (ms: number) => Promise<void>;
  cancelled?: () => boolean; now?: () => number; deadline: number; intervalMs: number; maxSteps: number;
  progress?: (value: T) => void | Promise<void>;
}): Promise<T | undefined> {
  const now = options.now ?? Date.now;
  if (!Number.isFinite(options.deadline) || !Number.isInteger(options.maxSteps) || options.maxSteps < 1 || options.maxSteps > 120 || !Number.isFinite(options.intervalMs) || options.intervalMs < 1000) throw new Error("Invalid bounded watch.");
  let last: T | undefined;
  for (let i = 0; i < options.maxSteps; i++) {
    if (options.cancelled?.() || now() >= options.deadline) break;
    last = await options.step();
    await options.progress?.(last);
    if (options.done(last) || options.cancelled?.() || now() >= options.deadline) break;
    if (i + 1 < options.maxSteps) await options.sleep(Math.min(options.intervalMs, Math.max(0, options.deadline - now())));
  }
  return last;
}
