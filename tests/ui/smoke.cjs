// Run with: NODE_PATH=/path/to/node_modules node tests/ui/smoke.cjs
// Requires Playwright and its Chromium browser; all /chat responses are mocked.
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH || undefined,
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 900 },
    });
    let calls = 0,
      mode = "success";
    await page.route("http://legal.test/", (route) =>
      route.fulfill({
        contentType: "text/html",
        body: fs.readFileSync(
          path.join(__dirname, "../../api/templates/index.html"),
          "utf8",
        ),
      }),
    );
    await page.route("http://legal.test/chat", async (route) => {
      calls++;
      const current = mode;
      if (current === "slow") {
        await new Promise((resolve) => setTimeout(resolve, 700));
        return route.fulfill({ status: 500, body: "error" }).catch(() => {});
      }
      if (current === "http-error")
        return route.fulfill({ status: 503, body: "unavailable" });
      const citations = {
        1: {
          metadata: { title: `Văn bản ${calls}`, article: "Điều 1" },
          content: '<img src=x onerror="window.injected=true"> Nội dung nguồn',
        },
      };
      let text = "Trả lời [1]";
      if (current === "unsafe")
        text +=
          '<img src=x onerror="window.injected=true"><script>window.injected=true</script><a href="javascript:alert(1)">link</a>';
      const events = [
        { step: "context_ready", status: "done", data: { citations } },
        {
          step: "answer",
          status: "streaming",
          data: { chunk: text, citations },
        },
      ];
      if (current !== "truncated")
        events.push({
          step: "answer",
          status: "done",
          data: { text, citations },
        });
      await route.fulfill({
        contentType: "text/event-stream",
        body:
          events
            .map((event) => "data: " + JSON.stringify(event) + "\r\n\r\n")
            .join("") + "data: [DONE]\r\n\r\n",
      });
    });
    await page.goto("http://legal.test/");
    await page.waitForFunction(() => window.marked && window.DOMPurify);
    const send = async (text) => {
      await page.locator("#userInput").fill(text);
      await page.locator("#sendBtn").click();
    };
    const idle = () =>
      page.waitForFunction(() => !document.querySelector("#sendBtn").disabled);
    await send("Câu hỏi một");
    await idle();
    await send("Câu hỏi hai");
    await idle();
    await page.locator(".citation").first().click();
    assert.match(await page.locator("#docList").innerText(), /Văn bản 1/);
    assert.equal(await page.locator("#docList img").count(), 0);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#openSources").click();
    assert(await page.locator("#sources").isVisible());
    assert(await page.locator("#main").evaluate((node) => node.inert));
    await page.keyboard.press("Escape");
    assert(!(await page.locator("#sources").isVisible()));
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    mode = "slow";
    await send("Câu hỏi chậm");
    await page.locator("#userInput").fill("Không gửi trùng");
    await page.locator("#userInput").press("Enter");
    assert.equal(calls, 3);
    await page.locator("#stopBtn").click();
    await idle();
    assert.match(await page.locator(".notice").last().innerText(), /Đã dừng/);
    mode = "unsafe";
    await page.getByRole("button", { name: "Thử lại", exact: true }).click();
    await idle();
    assert.equal(
      await page
        .locator(
          '.answer img,.answer script,.answer [onerror],.answer a[href^="javascript:"]',
        )
        .count(),
      0,
    );
    assert.equal(await page.evaluate(() => window.injected), undefined);
    assert.equal(await page.locator(".user").count(), 3);
    mode = "http-error";
    await send("Lỗi HTTP");
    await idle();
    assert.match(await page.locator(".notice").last().innerText(), /503/);
    mode = "truncated";
    await page.getByRole("button", { name: "Thử lại", exact: true }).click();
    await idle();
    assert.match(await page.locator(".notice").last().innerText(), /ngắt/);
    assert.match(await page.locator(".answer").last().innerText(), /Trả lời/);
    mode = "slow";
    await send("Reset khi đang chạy");
    await page.locator("#newChat").click();
    await page.waitForTimeout(900);
    assert.equal(await page.locator(".bot").count(), 0);
    assert.match(await page.locator("#docList").innerText(), /Chưa có nguồn/);
    await page.screenshot({ path: "/tmp/legal-ui-mobile.png" });
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.screenshot({ path: "/tmp/legal-ui-desktop.png" });
    console.log(
      "PASS: per-turn citations, mobile drawer, keyboard, duplicate guard, stop/retry, HTTP errors, interrupted SSE, reset race, HTML sanitization",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
