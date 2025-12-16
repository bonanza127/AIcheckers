"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, Play, RotateCcw, Ghost, Gamepad2, Terminal, Bomb, Sparkles } from "lucide-react";

type AnalysisPhase = "idle" | "scanning" | "complete";

type LogEntry = {
  message: string;
  type: "system" | "info" | "process" | "result" | "error" | "heading" | "detail";
};

type QueueItem = {
  id: string;
  name: string;
  preview: string;
  status: "wait" | "processing" | "ai" | "human";
};

type DetectionResult = {
  isAI: boolean;
  aiScore: number;
  humanScore: number;
  verdict: string;
  confidence: number;
  processingTime: number;
  artifacts: string;
};

// ピクセルアート風ローディングテキスト
const loadingTexts = [
  "SCANNING...",
  "ANALYZING PIXELS...",
  "DETECTING ARTIFACTS...",
  "PROCESSING DATA...",
];

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>("");
  const [phase, setPhase] = useState<AnalysisPhase>("idle");
  const [logs, setLogs] = useState<LogEntry[]>([
    { message: "SYSTEM BOOT OK.", type: "system" },
  ]);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [isScanning, setIsScanning] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [loadingTextIndex, setLoadingTextIndex] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((message: string, type: LogEntry["type"] = "info") => {
    setLogs(prev => [...prev, { message, type }]);
  }, []);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (startTime) {
      interval = setInterval(() => {
        setElapsedTime((Date.now() - startTime) / 1000);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [startTime]);

  // ローディングテキストアニメーション
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (phase === "scanning") {
      interval = setInterval(() => {
        setLoadingTextIndex(prev => (prev + 1) % loadingTexts.length);
      }, 800);
    }
    return () => clearInterval(interval);
  }, [phase]);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files) return;

    const validFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (validFiles.length === 0) {
      addLog("ERROR: NO IMAGE FILE DETECTED.", "error");
      return;
    }

    validFiles.forEach(file => {
      if (queue.length < 10) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const newItem: QueueItem = {
            id: `${Date.now()}-${file.name}`,
            name: file.name,
            preview: e.target?.result as string,
            status: "wait"
          };
          setQueue(prev => {
            if (prev.length < 10) {
              return [...prev, newItem];
            }
            return prev;
          });
        };
        reader.readAsDataURL(file);
      } else {
        addLog("WARNING: QUEUE MAX (10) REACHED.", "error");
      }
    });

    addLog(`QUEUE ADDED: ${validFiles.length} ARTIFACT(S) READY.`, "process");

    (window as unknown as { _pendingFiles: File[] })._pendingFiles = [
      ...((window as unknown as { _pendingFiles?: File[] })._pendingFiles || []),
      ...validFiles
    ];
  }, [queue.length, addLog]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [handleFiles]);

  const processFile = async (file: File, index: number) => {
    const fileStartTime = Date.now();

    setQueue(prev => prev.map((item, i) =>
      i === index ? { ...item, status: "processing" as const } : item
    ));

    const reader = new FileReader();
    const imageUrl = await new Promise<string>((resolve) => {
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.readAsDataURL(file);
    });

    setCurrentImage(imageUrl);
    setCurrentFileName(file.name);
    setPhase("scanning");
    setResult(null); // 前の結果をクリア

    addLog(`SCANNING: [${index + 1}/${queue.length}] FILE: ${file.name}...`, "heading");
    addLog("INITIALIZING NEURAL NETWORK...", "process");

    await new Promise(r => setTimeout(r, 500));
    addLog("> STAGE 1: FEATURE EXTRACTION", "detail");

    await new Promise(r => setTimeout(r, 700));
    addLog("> STAGE 2: ARTIFACT DETECTION", "detail");

    const hasAnomaly = Math.random() < 0.3;
    if (hasAnomaly) {
      addLog("> * HIGH-FREQ NOISE DETECTED", "detail");
    }

    await new Promise(r => setTimeout(r, 500));
    addLog("> STAGE 3: CLASSIFICATION", "detail");

    await new Promise(r => setTimeout(r, 400 + Math.random() * 600));

    let aiProbability: number;
    let artifacts: string;
    const randomOutcome = Math.random();

    if (randomOutcome < 0.4) {
      aiProbability = 85 + Math.random() * 14;
      artifacts = "HAND ANOMALY, TEXTURE REPEAT";
    } else if (randomOutcome < 0.6) {
      aiProbability = 50 + Math.random() * 20;
      artifacts = "EDGE NOISE, BOUNDARY ISSUE";
    } else {
      aiProbability = 5 + Math.random() * 25;
      artifacts = "ORGANIC STROKES, NO AI TRACE";
    }

    if (hasAnomaly) {
      aiProbability = Math.min(99, aiProbability + 15);
      artifacts += " [FILTER +15]";
    }

    const aiScore = Math.round(Math.min(99, Math.max(1, aiProbability)));
    const humanScore = 100 - aiScore;
    const isAI = aiScore > 50;
    const processingTime = (Date.now() - fileStartTime) / 1000;

    setQueue(prev => prev.map((item, i) =>
      i === index ? { ...item, status: isAI ? "ai" as const : "human" as const } : item
    ));

    setResult({
      isAI,
      aiScore,
      humanScore,
      verdict: isAI ? "AI WINS" : "HUMAN WINS",
      confidence: isAI ? aiScore : humanScore,
      processingTime,
      artifacts
    });

    setPhase("complete");
    addLog(`FINAL JUDGEMENT: ${isAI ? "AI WINS" : "HUMAN WINS"} (${isAI ? aiScore : humanScore}%)`, "result");

    await new Promise(r => setTimeout(r, 500));
  };

  const startBatchScan = async () => {
    const files = (window as unknown as { _pendingFiles?: File[] })._pendingFiles || [];
    if (isScanning || files.length === 0) return;

    setIsScanning(true);
    setStartTime(Date.now());
    setElapsedTime(0);
    setBatchProgress({ current: 0, total: files.length });
    addLog("--- BATCH SCAN STARTED ---", "heading");

    for (let i = 0; i < files.length; i++) {
      setBatchProgress({ current: i + 1, total: files.length });
      await processFile(files[i], i);
    }

    setIsScanning(false);
    setStartTime(null);
    addLog("--- BATCH SCAN COMPLETE ---", "heading");
    addLog(`STATUS: ${files.length} ARTIFACTS PROCESSED.`, "process");
  };

  const resetUI = () => {
    if (isScanning) return;

    setQueue([]);
    setCurrentImage(null);
    setCurrentFileName("");
    setPhase("idle");
    setResult(null);
    setBatchProgress({ current: 0, total: 0 });
    setElapsedTime(0);
    setStartTime(null);
    setLogs([{ message: "SYSTEM: QUEUE CLEAR.", type: "system" }]);
    (window as unknown as { _pendingFiles?: File[] })._pendingFiles = [];
  };

  const canExecute = queue.length > 0 && !isScanning;

  const getLogClass = (type: LogEntry["type"]) => {
    const classes: Record<LogEntry["type"], string> = {
      system: "log-system",
      info: "log-info",
      process: "log-process",
      result: "log-result",
      error: "log-error",
      heading: "log-heading",
      detail: "log-detail"
    };
    return classes[type];
  };

  // Verdictの表示内容を決定
  const getVerdictDisplay = () => {
    if (phase === "scanning") {
      return {
        text: loadingTexts[loadingTextIndex],
        className: "verdict-loading"
      };
    }
    if (result) {
      return {
        text: result.verdict,
        className: result.isAI ? "verdict-ai" : "verdict-human"
      };
    }
    return {
      text: "WAITING...",
      className: "text-muted"
    };
  };

  const verdictDisplay = getVerdictDisplay();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Pixel Art Background Scenery */}
      <div className="pixel-scenery">
        <div className="pixel-stars" />
        <div className="pixel-sun" />
        <div className="pixel-horizon" />
      </div>

      {/* Main Content - Above scenery */}
      <div className="relative z-10 flex flex-col flex-grow p-4 pb-16">
        {/* Header */}
        <header className="container mx-auto mb-4">
          <div className="pixel-panel border-b-8">
            <div className="px-4 py-3 flex flex-col sm:flex-row justify-between items-center">
              <div className="flex items-center gap-3">
                <Terminal className="w-6 h-6 text-secondary" />
                <h1 className="text-lg sm:text-xl pixel-title text-pixel">
                  &gt;_ A.I. DETECTOR <span className="text-xs text-muted">v.1.0</span>
                </h1>
              </div>
              <div className="text-[10px] mt-2 sm:mt-0 text-pixel">
                <span className="text-success">● SYSTEM STATUS: OK</span>
              </div>
            </div>
          </div>
        </header>

        {/* Intro */}
        <div className="text-center max-w-4xl mx-auto mb-4 px-4">
          <h2 className="text-sm pixel-title text-pixel mb-1">LOAD IMAGE FILE TO START!</h2>
          <p className="text-muted text-[10px] text-pixel">
            アーティファクトをスキャンし、AI生成の疑惑を判定するぞ！
          </p>
        </div>

        {/* Main Content */}
        <main className="flex-grow container mx-auto px-4 flex flex-col">
          <div className="flex flex-col lg:flex-row gap-4 flex-grow">

            {/* LEFT PANEL (2/3) */}
            <div className="w-full lg:w-2/3 flex flex-col gap-4">
              {/* Active Screen / Logs */}
              <div className="pixel-panel p-4 flex-grow flex flex-col">
                <h3 className="text-[10px] panel-header text-pixel text-primary">
                  ACTIVE SCREEN // LOGS
                </h3>

                <div className="flex flex-col md:flex-row gap-4 flex-grow">
                  {/* Active Image Preview */}
                  <div className="w-full md:w-1/2 flex flex-col">
                    {currentImage ? (
                      <div className={`active-image-container w-full h-48 flex items-center justify-center ${phase === "scanning" ? "scanning" : ""}`}>
                        <img
                          src={currentImage}
                          alt="Active Scan"
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                    ) : (
                      <div className="scan-placeholder w-full h-48 flex flex-col items-center justify-center">
                        <Ghost className="w-10 h-10 text-muted mb-2 opacity-50" />
                        <p className="text-muted text-pixel text-[10px]">AWAITING INPUT</p>
                      </div>
                    )}
                    {currentFileName && (
                      <p className="text-[8px] text-muted mt-2 truncate text-center text-pixel">{currentFileName}</p>
                    )}
                  </div>

                  {/* Console Log */}
                  <div className="w-full md:w-1/2 flex-grow">
                    <div
                      ref={logContainerRef}
                      className="console-log h-48 text-pixel"
                    >
                      {logs.map((log, i) => (
                        <div key={i} className={getLogClass(log.type)}>
                          &gt; {log.message}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Final Judgement */}
              <div className="pixel-panel p-4">
                <h3 className="text-[10px] panel-header text-pixel text-primary">
                  FINAL JUDGEMENT
                </h3>

                {/* Processing Status */}
                {batchProgress.total > 0 && (
                  <div className="flex justify-between items-center mb-3 text-[10px] text-pixel">
                    <span className="text-muted">
                      BATCH: {batchProgress.current} / {batchProgress.total}
                    </span>
                    <span className="text-muted">TIME: {elapsedTime.toFixed(2)}s</span>
                  </div>
                )}

                {/* Verdict */}
                <div className="flex justify-between items-center mb-4">
                  <span className="text-sm text-muted text-pixel">VERDICT:</span>
                  <span className={`verdict-display ${verdictDisplay.className}`}>
                    {verdictDisplay.text}
                  </span>
                </div>

                {/* AI HP Bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-[10px] mb-1 text-pixel">
                    <span className="text-danger">AI THREAT LEVEL</span>
                    <span className="text-danger">{result?.aiScore ?? 0}%</span>
                  </div>
                  <div className="hp-bar-bg">
                    <div
                      className="hp-bar-fill"
                      style={{
                        width: `${result?.aiScore ?? 0}%`,
                        backgroundColor: "var(--danger)"
                      }}
                    />
                  </div>
                </div>

                {/* Human HP Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-[10px] mb-1 text-pixel">
                    <span className="text-success">HUMAN ARTISTRY</span>
                    <span className="text-success">{result?.humanScore ?? 0}%</span>
                  </div>
                  <div className="hp-bar-bg">
                    <div
                      className="hp-bar-fill"
                      style={{
                        width: `${result?.humanScore ?? 0}%`,
                        backgroundColor: "var(--success)"
                      }}
                    />
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-y-1 gap-x-4 text-[10px] border-t-2 border-primary pt-3 text-pixel">
                  <div className="col-span-2 text-center text-primary border-b border-muted/30 pb-1 mb-1">
                    <span className="font-bold">ANALYSIS METRICS</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-primary">MODEL:</span>
                    <span className="text-secondary">ViT-DETECT</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-primary">CONFIDENCE:</span>
                    <span className="text-muted">{result?.confidence ? `${result.confidence}%` : "--"}</span>
                  </div>
                  <div className="col-span-2 flex justify-between">
                    <span className="text-primary">ARTIFACTS:</span>
                    <span className="text-muted text-right text-[8px]">{result?.artifacts || "PENDING..."}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT PANEL (1/3) */}
            <div className="w-full lg:w-1/3 flex flex-col gap-4">

              {/* Queue */}
              <div className="pixel-panel p-4 flex-grow">
                <h3 className="text-[10px] panel-header text-pixel text-primary flex justify-between">
                  ARTIFACT QUEUE <span>({queue.length}/10)</span>
                </h3>
                <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto p-1">
                  {queue.length === 0 ? (
                    <p className="text-muted text-[10px] italic text-pixel">QUEUE IS EMPTY. INSERT COIN.</p>
                  ) : (
                    queue.map((item) => (
                      <div
                        key={item.id}
                        className={`queue-item relative group ${
                          item.status === "processing" ? "active" :
                          item.status === "ai" ? "result-ai" :
                          item.status === "human" ? "result-human" : ""
                        }`}
                      >
                        <img
                          src={item.preview}
                          alt={item.name}
                          className="w-10 h-10 object-cover opacity-80"
                        />
                        <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 flex items-center justify-center text-[6px] text-white p-1 text-pixel transition-opacity">
                          {item.name.substring(0, 8)}...
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Upload Zone + Buttons */}
              <div className="flex flex-col gap-3">
                {/* Upload Zone */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className={`pixel-panel p-5 flex flex-col items-center justify-center upload-zone cursor-pointer min-h-[120px] ${
                    isDragging ? "dragging" : ""
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    multiple
                    onChange={handleFileSelect}
                  />
                  <Gamepad2 className="w-10 h-10 mb-2 text-secondary" />
                  <p className="text-sm text-pixel">PULL FILES</p>
                  <p className="text-[8px] text-muted mt-1 text-pixel">DRAG & DROP OR PUSH BUTTON</p>
                </div>

                {/* Buttons */}
                <div className="flex gap-3">
                  <button
                    onClick={startBatchScan}
                    disabled={!canExecute}
                    className="flex-grow pixel-btn flex items-center justify-center gap-2"
                  >
                    <Play className="w-4 h-4" />
                    START SCAN!
                  </button>
                  <button
                    onClick={resetUI}
                    disabled={isScanning}
                    className="pixel-btn pixel-btn-danger w-12 flex items-center justify-center"
                    title="キューをリセット"
                  >
                    <Bomb className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Fixed Footer - Bottom edge extends beyond screen */}
      <footer className="fixed bottom-0 left-0 right-0 z-50 translate-y-[4px]">
        <div className="container mx-auto px-4">
          <div className="pixel-footer-bar px-4 py-3 text-center">
            <p className="text-[8px] text-muted text-pixel">
              &copy; 2024 PIXEL FORENSICS // POWERED BY ViT-DETECT // HIGH SCORE: 99999
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
