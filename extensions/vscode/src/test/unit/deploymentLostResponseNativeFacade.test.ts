import assert from "node:assert/strict";
import test from "node:test";
import { lostResponseNativeFacade } from "../helpers/deploymentLostResponseNativeFacade";

test("native fallback methods and accessors preserve private-field receivers, including detached methods", async () => {
  class NativeWebview {
    #html = "";
    #messages: unknown[] = [];
    get html() { return this.#html; }
    set html(value: string) { this.#html = value; }
    async postMessage(value: unknown) { this.#messages.push(value); return true; }
    get messages() { return [...this.#messages]; }
    onDidReceiveMessage(_listener: unknown) { return "native"; }
  }
  class NativePanel {
    #webview = new NativeWebview();
    #disposed = false;
    get webview() { return this.#webview; }
    dispose() { this.#disposed = true; }
    get disposed() { return this.#disposed; }
  }
  const panel = new NativePanel(), native = panel.webview;
  const wrapped = lostResponseNativeFacade(native, { onDidReceiveMessage: () => "observed" });
  const facade = lostResponseNativeFacade(panel, { webview: wrapped });
  facade.webview.html = "production HTML";
  assert.equal(native.html, "production HTML"); assert.equal(facade.webview.html, native.html);
  const detached = facade.webview.postMessage;
  assert.equal(await detached({ kind: "busy", value: true }), true);
  assert.deepEqual(native.messages, [{ kind: "busy", value: true }]);
  assert.equal(facade.webview.onDidReceiveMessage(undefined), "observed");
  facade.dispose(); assert.equal(panel.disposed, true);
});
