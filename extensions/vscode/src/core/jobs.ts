const yamlAPIVersion = /^\s*apiVersion\s*:\s*["']?agefreighter\.io\/v2["']?\s*(?:#.*)?$/m;
const yamlKind = /^\s*kind\s*:\s*["']?LoadJob["']?\s*(?:#.*)?$/m;

export function looksLikeAgefreighterJob(data: Uint8Array): boolean {
  if (data.byteLength === 0 || data.byteLength > 1024 * 1024) {
    return false;
  }
  const text = new TextDecoder("utf-8", { fatal: false }).decode(data);
  const trimmed = text.trimStart();
  if (trimmed.startsWith("{")) {
    try {
      const parsed = JSON.parse(text) as Record<string, unknown>;
      return parsed.apiVersion === "agefreighter.io/v2" && parsed.kind === "LoadJob";
    } catch {
      return false;
    }
  }
  return yamlAPIVersion.test(text) && yamlKind.test(text);
}
