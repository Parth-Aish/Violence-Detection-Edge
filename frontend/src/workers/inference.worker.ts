/**
 * inference.worker.ts
 * Legacy Web Worker shim — now re-exports EdgeEngine for use as a Worker.
 * The new architecture uses edgeEngine singleton directly on the main thread
 * via requestAnimationFrame (see page.tsx). This file is kept to avoid
 * breaking any existing import, but is no longer the primary inference path.
 */

// No-op — inference is now handled by edgeEngine.ts on the main thread
export {};
