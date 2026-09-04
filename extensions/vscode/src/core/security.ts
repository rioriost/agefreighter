const sensitiveKey = /(password|secret|token|credential|connection|dsn|uri|query|path)/i;
const urlCredential = /([a-z][a-z0-9+.-]*:\/\/)([^\s/@:]+):([^\s/@]+)@/gi;

export interface RedactionLimits {
  maxDepth: number;
  maxArrayItems: number;
  maxStringLength: number;
  maxSerializedBytes: number;
}

export const defaultRedactionLimits: RedactionLimits = {
  maxDepth: 8,
  maxArrayItems: 100,
  maxStringLength: 2000,
  maxSerializedBytes: 256 * 1024
};

export function redactText(value: string, maxLength = 4000): string {
  const withoutCredentials = value.replace(urlCredential, "$1[redacted]@");
  if (withoutCredentials.length <= maxLength) {
    return withoutCredentials;
  }
  return `${withoutCredentials.slice(0, maxLength)}\n[truncated]`;
}

export function redactForModel(
  value: unknown,
  limits: RedactionLimits = defaultRedactionLimits
): unknown {
  const seen = new WeakSet<object>();

  const visit = (current: unknown, depth: number): unknown => {
    if (depth > limits.maxDepth) {
      return "[depth limit]";
    }
    if (typeof current === "string") {
      return redactText(current, limits.maxStringLength);
    }
    if (current === null || typeof current !== "object") {
      return current;
    }
    if (seen.has(current)) {
      return "[cycle]";
    }
    seen.add(current);
    if (Array.isArray(current)) {
      const values = current.slice(0, limits.maxArrayItems).map((item) => visit(item, depth + 1));
      if (current.length > limits.maxArrayItems) {
        values.push(`[${current.length - limits.maxArrayItems} more items]`);
      }
      return values;
    }
    const output: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(current)) {
      output[key] = sensitiveKey.test(key) ? "[redacted]" : visit(item, depth + 1);
    }
    return output;
  };

  const redacted = visit(value, 0);
  const serialized = JSON.stringify(redacted);
  if (Buffer.byteLength(serialized, "utf8") > limits.maxSerializedBytes) {
    return {
      truncated: true,
      reason: "Bounded AGEFreighter evidence exceeded the model context limit. Open the full local report in VS Code."
    };
  }
  return redacted;
}
