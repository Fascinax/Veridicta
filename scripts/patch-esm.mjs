/**
 * Patch vscode-jsonrpc to add the missing ESM subpath export "./node".
 *
 * @github/copilot-sdk imports "vscode-jsonrpc/node" (without .js), which Node.js
 * v22+ strict-ESM mode can't resolve because vscode-jsonrpc has no exports map.
 * This script runs once after `npm install` (via the postinstall hook) to add the
 * required exports field so Node.js resolves the subpath correctly.
 */

import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgPath = join(__dirname, "..", "node_modules", "vscode-jsonrpc", "package.json");

const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));

if (!pkg.exports) {
  pkg.exports = {
    ".": {
      require: "./lib/node/main.js",
      import: "./lib/node/main.js",
      default: "./lib/node/main.js",
    },
    "./node": {
      require: "./node.js",
      import: "./node.js",
      default: "./node.js",
    },
    "./node.js": {
      require: "./node.js",
      import: "./node.js",
      default: "./node.js",
    },
    "./browser": {
      require: "./browser.js",
      import: "./browser.js",
      default: "./browser.js",
    },
  };
  writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
  console.log("patched vscode-jsonrpc: added ESM exports map for ./node subpath");
} else {
  console.log("vscode-jsonrpc already has exports, skipping patch");
}
