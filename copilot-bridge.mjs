/**
 * Node.js bridge for Veridicta — GitHub Copilot LLM backend.
 *
 * Uses @github/copilot-sdk for proper Copilot authentication and chat completions.
 *
 * Usage:  node copilot-bridge.mjs [model] [--stream]
 * Reads JSON from stdin:  { "system": "...", "user": "..." }
 *
 * Non-streaming (default):
 *   Outputs single JSON to stdout: { "content": "..." }
 *
 * Streaming (--stream flag):
 *   Outputs JSONL lines to stdout: { "partial": "<deltaContent>" }
 *   One line per token chunk. EOF signals end of stream.
 *
 * Authentication: GITHUB_PAT env var, or `gh auth login` fallback.
 */

import { CopilotClient, approveAll } from "@github/copilot-sdk";

const model = process.argv[2] || "gpt-4.1";
const streamMode = process.argv.includes("--stream");

const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}
const input = JSON.parse(Buffer.concat(chunks).toString("utf-8"));

const token =
  process.env.GITHUB_PAT ||
  process.env.COPILOT_GITHUB_TOKEN ||
  process.env.GH_TOKEN ||
  process.env.GITHUB_TOKEN;

const client = new CopilotClient({
  githubToken: token || undefined,
  useLoggedInUser: !token,
});

/**
 * Extract text from the SDK response, handling multiple possible shapes.
 */
function extractText(obj) {
  if (!obj) return "";
  if (typeof obj === "string") return obj;

  if (obj.content && typeof obj.content === "string") return obj.content;
  if (obj.data?.content && typeof obj.data.content === "string")
    return obj.data.content;
  if (obj.text && typeof obj.text === "string") return obj.text;

  if (Array.isArray(obj)) {
    return obj.map(extractText).filter(Boolean).join("\n");
  }
  if (Array.isArray(obj.messages)) {
    return obj.messages
      .filter((m) => m.role === "assistant" || m.role === "model")
      .map((m) => m.content || m.text || "")
      .filter(Boolean)
      .join("\n");
  }
  if (obj.choices && Array.isArray(obj.choices)) {
    return obj.choices
      .map((c) => c.message?.content || c.text || "")
      .filter(Boolean)
      .join("\n");
  }

  return JSON.stringify(obj);
}

try {
  const session = await client.createSession({
    model,
    systemMessage: input.system ? { content: input.system } : undefined,
    tools: [],
    onPermissionRequest: approveAll,
  });

  if (streamMode) {
    // Stream mode: subscribe to incremental deltas, output as JSONL
    session.on("assistant.message_delta", (event) => {
      const delta = event.data.deltaContent;
      if (delta) {
        process.stdout.write(JSON.stringify({ partial: delta }) + "\n");
      }
    });

    // sendAndWait waits for session.idle; delta handlers fire during this
    await session.sendAndWait({ prompt: input.user }, 180_000);
  } else {
    // Non-streaming mode: output single JSON object
    const response = await session.sendAndWait(
      { prompt: input.user },
      180_000,
    );

    const text = extractText(response);

    if (!text) {
      process.stderr.write(
        JSON.stringify({
          debug: "empty_response",
          type: typeof response,
          keys:
            response && typeof response === "object"
              ? Object.keys(response)
              : null,
        }) + "\n",
      );
    }

    process.stdout.write(JSON.stringify({ content: text }));
  }

  await session.destroy?.();
  await client.stop();
  process.exit(0);
} catch (err) {
  process.stderr.write(JSON.stringify({ error: err.message }));
  await client.stop?.();
  process.exit(1);
}
