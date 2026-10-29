/**
 * edgeDetector.ts
 * Unified client-side video ingestion and canvas pipeline.
 * Supports: webcam, local file, preset sample video.
 */

export type VideoSource = "webcam" | "file" | "sample" | "idle";

export interface EdgeDetectorCallbacks {
  onFrame: (
    frameData: ImageData,
    frameIndex: number,
    prevFrameData: ImageData | null,
  ) => void;
  onError: (message: string) => void;
  onStop: () => void;
}

export class EdgeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private videoEl: HTMLVideoElement;
  private source: VideoSource = "idle";
  private running = false;
  private frameIndex = 0;
  private prevFrameData: ImageData | null = null;
  private callbacks: EdgeDetectorCallbacks;
  private animId: number | null = null;
  private mediaStream: MediaStream | null = null;
  private lastFrameTime = 0;
  private targetFps = 30;

  constructor(callbacks: EdgeDetectorCallbacks) {
    this.callbacks = callbacks;
    this.canvas = document.createElement("canvas");
    this.canvas.width = 320;
    this.canvas.height = 320;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.videoEl = document.createElement("video");
    this.videoEl.muted = true;
    this.videoEl.playsInline = true;
    this.videoEl.loop = true;
  }

  get currentSource(): VideoSource { return this.source; }
  get isRunning(): boolean { return this.running; }

  async startWebcam(): Promise<void> {
    await this.stop();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "environment" },
      audio: false,
    });
    this.mediaStream = stream;
    this.videoEl.srcObject = stream;
    await this.videoEl.play();
    this.source = "webcam";
    this._startLoop();
  }

  async startSample(url: string): Promise<void> {
    await this.stop();
    this.videoEl.srcObject = null;
    this.videoEl.src = url;
    this.videoEl.loop = true;
    try {
      await this.videoEl.play();
    } catch {
      // autoplay may require interaction; show UI hint
    }
    this.source = "sample";
    this._startLoop();
  }

  async startFile(file: File): Promise<void> {
    await this.stop();
    const url = URL.createObjectURL(file);
    this.videoEl.srcObject = null;
    this.videoEl.src = url;
    this.videoEl.loop = true;
    await this.videoEl.play();
    this.source = "file";
    this._startLoop();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    this.videoEl.pause();
    this.videoEl.src = "";
    this.videoEl.srcObject = null;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }
    this.source = "idle";
    this.prevFrameData = null;
    this.frameIndex = 0;
    this.callbacks.onStop();
  }

  private _startLoop() {
    this.running = true;
    this.lastFrameTime = 0;
    this.frameIndex = 0;

    const loop = (now: number) => {
      if (!this.running) return;
      this.animId = requestAnimationFrame(loop);

      const elapsed = now - this.lastFrameTime;
      if (elapsed < 1000 / this.targetFps) return;
      this.lastFrameTime = now;

      if (this.videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

      this.ctx.drawImage(this.videoEl, 0, 0, 320, 320);
      const frameData = this.ctx.getImageData(0, 0, 320, 320);
      this.callbacks.onFrame(frameData, this.frameIndex, this.prevFrameData);
      this.prevFrameData = frameData;
      this.frameIndex++;
    };

    this.animId = requestAnimationFrame(loop);
  }

  /** Expose the off-screen canvas for overlay rendering */
  getCanvas(): HTMLCanvasElement { return this.canvas; }
}
