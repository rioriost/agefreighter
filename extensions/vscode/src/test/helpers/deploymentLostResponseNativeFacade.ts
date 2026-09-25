/** VS Code's concrete native objects may use JavaScript private fields.
 * Keep their original receiver for fallback methods/accessors; namespace-only
 * facades are insufficient for Webview, WebviewPanel and similar instances. */
export function lostResponseNativeFacade<T extends object>(original: T, overrides: Partial<T>): T {
  return new Proxy({} as T, {
    get: (_target, key) => {
      if (Object.hasOwn(overrides, key)) return Reflect.get(overrides, key);
      const value = Reflect.get(original, key, original);
      return typeof value === "function" ? value.bind(original) : value;
    },
    set: (_target, key, value) => Reflect.set(original, key, value, original)
  });
}
