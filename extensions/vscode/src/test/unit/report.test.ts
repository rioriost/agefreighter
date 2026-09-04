import assert from "node:assert/strict";
import test from "node:test";
import { escapeHTML, reportHTML } from "../../core/report";

test("escapes every HTML-significant character", () => {
  assert.equal(
    escapeHTML(`<script data-x="'">&</script>`),
    "&lt;script data-x=&quot;&#39;&quot;&gt;&amp;&lt;/script&gt;"
  );
});

test("report webview disables scripts and escapes evidence", () => {
  const html = reportHTML("unsafe <title>", { value: "</pre><script>alert(1)</script>" });
  assert.match(html, /default-src 'none'/);
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.match(html, /unsafe &lt;title&gt;/);
});
