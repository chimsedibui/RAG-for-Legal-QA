# UI smoke test

Tests use mocked `/chat` responses and never contact an LLM. They cover citations
across turns, the mobile source drawer, duplicate submission, cancellation/retry,
HTTP errors, incomplete streams, reset races and unsafe HTML. Markdown and
DOMPurify load from the same CDN URLs used by the application, so Internet access
is needed. Missing libraries fail the test; the application falls back to plain
text if these libraries cannot load.

Install test tooling outside the repository if desired:

```bash
npm install --prefix /tmp/legal-ui-tests playwright
/tmp/legal-ui-tests/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/legal-ui-tests/node_modules node tests/ui/smoke.cjs
```

Set `CHROMIUM_PATH` to use an existing Chromium executable. Screenshots are saved
as `/tmp/legal-ui-mobile.png` and `/tmp/legal-ui-desktop.png`.

The Stop button aborts the browser request. The server closes its pipeline at the
next yielded event; a blocking upstream model request may finish before this
cancellation takes effect. Source links appear only when source metadata supplies
an HTTP(S) URL; the UI does not invent original-document URLs.
