import { randomBytes } from "node:crypto";

export function escapeHTML(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function reportHTML(title: string, document: unknown): string {
  const nonce = randomBytes(16).toString("base64");
  const body = escapeHTML(JSON.stringify(document, null, 2));
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${nonce}'">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHTML(title)}</title>
  <style nonce="${nonce}">
    body { color: var(--vscode-editor-foreground); background: var(--vscode-editor-background); padding: 1rem 1.25rem; }
    h1 { font: 600 1.25rem var(--vscode-font-family); margin: 0 0 1rem; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; font: var(--vscode-editor-font-size) var(--vscode-editor-font-family); }
  </style>
</head>
<body>
  <h1>${escapeHTML(title)}</h1>
  <pre>${body}</pre>
</body>
</html>`;
}
