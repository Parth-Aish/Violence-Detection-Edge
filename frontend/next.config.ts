import fs from "node:fs";
import path from "node:path";
import type { NextConfig } from "next";

/**
 * Enterprise Dual-Engine Build Step:
 * Ensure .wasm and .mjs binaries from onnxruntime-web are copied
 * to public build directories (public/ort-wasm and public/) so Edge AI WASM inference
 * runs client-side with zero missing assets.
 */
function syncOnnxWasmBinaries(): void {
  try {
    const ortDist = path.resolve(process.cwd(), "node_modules", "onnxruntime-web", "dist");
    const publicWasmDir = path.resolve(process.cwd(), "public", "ort-wasm");
    const publicDir = path.resolve(process.cwd(), "public");

    if (fs.existsSync(ortDist)) {
      if (!fs.existsSync(publicWasmDir)) {
        fs.mkdirSync(publicWasmDir, { recursive: true });
      }
      const files = fs.readdirSync(ortDist);
      for (const file of files) {
        if (file.endsWith(".wasm") || file.endsWith(".mjs")) {
          const src = path.join(ortDist, file);
          const dest = path.join(publicWasmDir, file);
          if (!fs.existsSync(dest) || fs.statSync(src).size !== fs.statSync(dest).size) {
            fs.copyFileSync(src, dest);
          }
          if (file.endsWith(".wasm")) {
            const rootDest = path.join(publicDir, file);
            if (!fs.existsSync(rootDest) || fs.statSync(src).size !== fs.statSync(rootDest).size) {
              fs.copyFileSync(src, rootDest);
            }
          }
        }
      }
    }
  } catch (err) {
    console.warn("[next.config.ts] Warning syncing onnxruntime-web WASM files:", err);
  }
}

syncOnnxWasmBinaries();

const nextConfig: NextConfig = {
  output: "export",

  /**
   * COOP + COEP headers required for SharedArrayBuffer,
   * which onnxruntime-web needs for multi-threaded WASM.
   */
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy",  value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },

  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },

  webpack(config) {
    syncOnnxWasmBinaries();
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false };
    return config;
  },
};

export default nextConfig;
