/** Lazy test-only namespace facade. Never enumerate/evaluate proposed API
 * getters, and keep TypeScript's __importStar from copying the empty target. */
export function lazyNativeFacade<T extends object>(original: T, overrides: Partial<T>): T {
  return new Proxy({} as T, { get: (_target, name) => name === "__esModule" ? true :
    Object.hasOwn(overrides, name) ? Reflect.get(overrides, name) : Reflect.get(original, name) });
}
