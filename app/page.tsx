"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const MODEL_URL = "/yolo11n_pose_7class_v1.onnx";
const IMG_SIZE = 640;
const NUM_CLASSES = 7;
const NUM_KPTS = 4;
const CONF_THRES = 0.35;
const IOU_THRES = 0.45;

const CLASS_NAMES = [
  "ID_Card_Front",
  "ID_Card_Back",
  "New_Card_Front",
  "New_Car_Back",
  "Old_Card_Front",
  "Old_Card_Back",
  "Other_Card"
];

const KPT_NAMES = ["TL", "TR", "BR", "BL"];

type Detection = {
  box: [number, number, number, number];
  score: number;
  cls: number;
  kpts: { x: number; y: number; conf: number }[];
};

type LetterboxInfo = {
  scale: number;
  padX: number;
  padY: number;
  origW: number;
  origH: number;
};

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function iou(a: Detection, b: Detection) {
  const [ax1, ay1, ax2, ay2] = a.box;
  const [bx1, by1, bx2, by2] = b.box;

  const x1 = Math.max(ax1, bx1);
  const y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);

  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);

  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(dets: Detection[]) {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];

  while (sorted.length > 0) {
    const best = sorted.shift()!;
    keep.push(best);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(best, sorted[i]) > IOU_THRES) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}

function preprocessVideo(video: HTMLVideoElement) {
  const origW = video.videoWidth;
  const origH = video.videoHeight;

  const scale = Math.min(IMG_SIZE / origW, IMG_SIZE / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);

  const padX = Math.floor((IMG_SIZE - newW) / 2);
  const padY = Math.floor((IMG_SIZE - newH) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;

  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  ctx.drawImage(video, 0, 0, origW, origH, padX, padY, newW, newH);

  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const pixels = imageData.data;

  const input = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);

  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const pixelIndex = (y * IMG_SIZE + x) * 4;

      const r = pixels[pixelIndex] / 255;
      const g = pixels[pixelIndex + 1] / 255;
      const b = pixels[pixelIndex + 2] / 255;

      const hw = y * IMG_SIZE + x;

      input[hw] = r;
      input[IMG_SIZE * IMG_SIZE + hw] = g;
      input[2 * IMG_SIZE * IMG_SIZE + hw] = b;
    }
  }

  const tensor = new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);

  return {
    tensor,
    info: { scale, padX, padY, origW, origH },
  };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const runningRef = useRef(false);
  const busyRef = useRef(false);
  const needsSigmoidRef = useRef<boolean | null>(null);

  const [status, setStatus] = useState("Loading model...");
  const [backend, setBackend] = useState("loading");
  const [fps, setFps] = useState<number | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);

  useEffect(() => {
    async function parseOutput(output: ort.Tensor, info: LetterboxInfo): Promise<Detection[]> {
      let data: Float32Array;
      const rawData = output.data;
      if (rawData instanceof Promise) {
        data = (await rawData) as Float32Array;
      } else {
        data = rawData as Float32Array;
      }

      const dims = output.dims;
      if (dims.length !== 3) {
        throw new Error(`Unexpected output shape: ${dims.join(",")}`);
      }

      let channels: number;
      let anchors: number;
      let get: (c: number, a: number) => number;

      // Common YOLO shapes: [1, C, N] or [1, N, C]
      if (dims[1] < dims[2]) {
        channels = dims[1];
        anchors = dims[2];
        get = (c, a) => data[c * anchors + a];
      } else {
        anchors = dims[1];
        channels = dims[2];
        get = (c, a) => data[a * channels + c];
      }

      const expectedChannels = 4 + NUM_CLASSES + NUM_KPTS * 3;
      if (channels !== expectedChannels) {
        console.warn(`Expected ${expectedChannels} channels, got ${channels}. Dims: ${dims.join(",")}`);
      }

      // Auto-detect if scores are raw logits or already probabilities
      if (needsSigmoidRef.current === null) {
        let foundNegative = false;
        for (let a = 0; a < Math.min(anchors, 100); a++) {
          for (let c = 0; c < NUM_CLASSES; c++) {
            if (get(4 + c, a) < 0) {
              foundNegative = true;
              break;
            }
          }
          if (foundNegative) break;
        }
        needsSigmoidRef.current = foundNegative;
        console.log(`Auto-detected class scores: ${foundNegative ? "Logits (using Sigmoid)" : "Probabilities"}`);
      }

      const detections: Detection[] = [];

      for (let a = 0; a < anchors; a++) {
        let bestCls = 0;
        let bestScore = -Infinity;

        for (let c = 0; c < NUM_CLASSES; c++) {
          let score = get(4 + c, a);
          if (needsSigmoidRef.current) {
            score = sigmoid(score);
          }

          if (score > bestScore) {
            bestScore = score;
            bestCls = c;
          }
        }

        if (bestScore < CONF_THRES) continue;

        const cx = get(0, a);
        const cy = get(1, a);
        const w = get(2, a);
        const h = get(3, a);

        let x1 = cx - w / 2;
        let y1 = cy - h / 2;
        let x2 = cx + w / 2;
        let y2 = cy + h / 2;

        // Scale and shift back using letterbox padding and scale
        x1 = (x1 - info.padX) / info.scale;
        y1 = (y1 - info.padY) / info.scale;
        x2 = (x2 - info.padX) / info.scale;
        y2 = (y2 - info.padY) / info.scale;

        const kpts = [];
        const kptBase = 4 + NUM_CLASSES;

        for (let k = 0; k < NUM_KPTS; k++) {
          let kx = get(kptBase + k * 3, a);
          let ky = get(kptBase + k * 3 + 1, a);
          let kc = get(kptBase + k * 3 + 2, a);

          if (needsSigmoidRef.current) {
            kc = sigmoid(kc);
          }

          kx = (kx - info.padX) / info.scale;
          ky = (ky - info.padY) / info.scale;

          kpts.push({
            x: clamp(kx, 0, info.origW),
            y: clamp(ky, 0, info.origH),
            conf: kc,
          });
        }

        detections.push({
          box: [
            clamp(x1, 0, info.origW),
            clamp(y1, 0, info.origH),
            clamp(x2, 0, info.origW),
            clamp(y2, 0, info.origH),
          ],
          score: bestScore,
          cls: bestCls,
          kpts,
        });
      }

      return nms(detections);
    }

    function drawDetections(canvas: HTMLCanvasElement, detections: Detection[]) {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (detections.length === 0) return;

      // Only draw the single best detection
      const det = detections[0];
      const [x1, y1, x2, y2] = det.box;

      // Bounding Box (Green)
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#10B981"; // Emerald green
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label class name and confidence
      ctx.font = "bold 16px sans-serif";
      ctx.fillStyle = "#10B981";
      ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
      ctx.shadowBlur = 4;
      ctx.fillText(
        `${CLASS_NAMES[det.cls]} ${(det.score * 100).toFixed(1)}%`,
        x1,
        Math.max(20, y1 - 8)
      );
      ctx.shadowBlur = 0;

      const pts = det.kpts;

      if (pts.length >= 4) {
        // Magenta polygon connecting TL -> TR -> BR -> BL
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#EC4899"; // Magenta
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y); // TL
        ctx.lineTo(pts[1].x, pts[1].y); // TR
        ctx.lineTo(pts[2].x, pts[2].y); // BR
        ctx.lineTo(pts[3].x, pts[3].y); // BL
        ctx.closePath();
        ctx.stroke();

        // Colored keypoints: TL (blue), TR (yellow), BR (red), BL (cyan)
        const colors = ["#3B82F6", "#FBBF24", "#EF4444", "#06B6D4"];
        for (let i = 0; i < 4; i++) {
          ctx.beginPath();
          ctx.fillStyle = colors[i];
          ctx.arc(pts[i].x, pts[i].y, 7, 0, Math.PI * 2);
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#FFFFFF";
          ctx.stroke();

          ctx.font = "bold 13px sans-serif";
          ctx.fillStyle = "#FFFFFF";
          ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
          ctx.shadowBlur = 4;
          ctx.fillText(KPT_NAMES[i], pts[i].x + 10, pts[i].y - 10);
          ctx.shadowBlur = 0;
        }
      }
    }

    async function loop() {
      if (!runningRef.current) return;

      const session = sessionRef.current;
      const video = videoRef.current;
      const overlay = overlayRef.current;

      if (!session || !video || !overlay || busyRef.current) {
        requestAnimationFrame(loop);
        return;
      }

      busyRef.current = true;
      const t0 = performance.now();

      try {
        const { tensor, info } = preprocessVideo(video);

        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const outputs = await session.run({ [inputName]: tensor });
        const output = outputs[outputName];

        console.log("ONNX output dims:", output.dims);

        const detections = await parseOutput(output, info);
        drawDetections(overlay, detections);

        const t1 = performance.now();
        setFps(1000 / Math.max(1, t1 - t0));
      } catch (err) {
        console.error("Error in inference loop:", err);
      }

      busyRef.current = false;
      requestAnimationFrame(loop);
    }

    async function init() {
      try {
        // Set WASM paths to unpkg or jsdelivr CDN matching onnxruntime-web structure
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

        let session: ort.InferenceSession;

        try {
          console.log("Attempting to load model with WebGPU...");
          session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all",
          });
          setBackend("webgpu");
          console.log("Successfully initialized WebGPU backend.");
        } catch (err) {
          console.warn("WebGPU failed. Falling back to WASM.", err);
          try {
            session = await ort.InferenceSession.create(MODEL_URL, {
              executionProviders: ["wasm"],
              graphOptimizationLevel: "all",
            });
            setBackend("wasm");
            console.log("Successfully initialized WASM backend.");
          } catch (wasmErr) {
            console.error("WASM fallback also failed.", wasmErr);
            throw new Error(`Failed to load model on both WebGPU and WASM: ${String(wasmErr)}`);
          }
        }

        sessionRef.current = session;

        console.log("Requesting camera stream...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        const video = videoRef.current;
        if (!video) throw new Error("Video element not found.");

        video.srcObject = stream;

        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
        });

        await video.play();

        const overlay = overlayRef.current;
        if (overlay) {
          overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;
        }

        runningRef.current = true;
        setStatus("Running");

        // Start loop
        requestAnimationFrame(loop);
      } catch (err) {
        console.error("Initialization error:", err);
        setStatus("Error");
        setErrorDetails(String(err));
      }
    }

    init();

    const currentVideo = videoRef.current;

    return () => {
      runningRef.current = false;
      if (currentVideo) {
        const stream = currentVideo.srcObject as MediaStream | null;
        stream?.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-between p-4 md:p-8 font-sans">
      <div className="w-full max-w-4xl flex flex-col flex-1">
        {/* Header */}
        <header className="mb-6 flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-800 pb-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 to-indigo-500 bg-clip-text text-transparent font-sans">
              Egyptian Card Corner Detection
            </h1>
            <p className="text-slate-400 text-xs md:text-sm mt-1">
              Real-time semantic keypoint and card boundary alignment via ONNX Web
            </p>
          </div>

          {/* Badges / Metrics */}
          <div className="mt-4 md:mt-0 flex flex-wrap gap-2">
            <span
              id="status-badge"
              className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${
                status === "Running"
                  ? "bg-emerald-950/50 border-emerald-500/30 text-emerald-400"
                  : status.startsWith("Error")
                  ? "bg-rose-950/50 border-rose-500/30 text-rose-400"
                  : "bg-amber-950/50 border-amber-500/30 text-amber-400"
              }`}
            >
              Status: {status}
            </span>

            <span
              id="backend-badge"
              className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${
                backend === "webgpu"
                  ? "bg-cyan-950/50 border-cyan-500/30 text-cyan-400 font-mono"
                  : backend === "wasm"
                  ? "bg-indigo-950/50 border-indigo-500/30 text-indigo-400 font-mono"
                  : "bg-slate-900 border-slate-700 text-slate-400"
              }`}
            >
              Backend: {backend}
            </span>

            <span
              id="fps-badge"
              className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border bg-slate-900 border-slate-700 text-slate-300 font-mono"
            >
              FPS: {fps !== null ? fps.toFixed(1) : "N/A"}
            </span>
          </div>
        </header>

        {/* Video Canvas Container */}
        <div className="relative w-full flex-1 flex items-center justify-center min-h-[300px] bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800 overflow-hidden shadow-2xl p-2">
          {errorDetails && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-6 text-center bg-slate-950/90">
              <svg className="w-12 h-12 text-rose-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <h3 className="text-lg font-bold text-rose-400 mb-1">Initialization Failed</h3>
              <p className="text-slate-400 text-sm max-w-md">{errorDetails}</p>
            </div>
          )}

          <div className="relative w-full max-w-2xl aspect-video rounded-xl overflow-hidden bg-black shadow-inner">
            <video
              ref={videoRef}
              className="block w-full h-full object-cover"
              playsInline
              muted
            />

            <canvas
              ref={overlayRef}
              className="absolute left-0 top-0 w-full h-full object-cover pointer-events-none"
            />
          </div>
        </div>

        {/* Notes/Semantic Rule explanation */}
        <footer className="mt-6 p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 text-xs md:text-sm text-slate-300">
          <div className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400 font-semibold border border-indigo-500/20">
              i
            </span>
            <div>
              <p className="font-semibold text-slate-100 mb-1">Important Semantic Note:</p>
              <p className="leading-relaxed">
                The corner detection output order is semantic:{" "}
                <span className="text-blue-400 font-semibold">TL (Top Left)</span> →{" "}
                <span className="text-yellow-400 font-semibold">TR (Top Right)</span> →{" "}
                <span className="text-red-400 font-semibold">BR (Bottom Right)</span> →{" "}
                <span className="text-cyan-400 font-semibold">BL (Bottom Left)</span>.
              </p>
              <p className="mt-1.5 text-slate-400">
                Do <strong>NOT</strong> sort the corners geometrically. This direct ordering is crucial for proper perspective warping and homography computation downstream.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
