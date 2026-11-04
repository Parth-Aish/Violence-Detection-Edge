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
        }
      }
    }
  } catch (err) {
    console.warn("[next.config.ts] Warning syncing onnxruntime-web WASM files:", err);
  }
}

syncOnnxWasmBinaries();

const isDev = process.env.NODE_ENV === "development";

const nextConfig: NextConfig = {
  ...(isDev
    ? {
        async rewrites() {
          return [
            {
              source: "/api/:path*",
              destination: "http://127.0.0.1:8000/api/:path*",
            },
          ];
        },
      }
    : {
        output: "export",
      }),

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
