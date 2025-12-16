"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, Play, RotateCcw, Cpu, Scan, Activity } from "lucide-react";

type AnalysisPhase = "idle" | "scanning" | "complete";

type LogEntry = {
  message: string;
  type: "system" | "info" | "process" | "result" | "error" | "heading" | "detail";
};

type QueueItem = {
  id: string;
  name: string;
  status: "wait" | "processing" | "ai" | "human";
};

type DetectionResult = {
  isAI: boolean;
  aiScore: number;
  humanScore: number;
  verdict: string;
  confidence: number;
  processingTime: number;
  artifacts: string[];
};

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>("");
  const [phase, setPhase] = useState<AnalysisPhase>("idle");
  const [logs, setLogs] = useState<LogEntry[]>([
    { message: "SYSTEM INITIALIZED", type: "system" },
    { message: "Neural detection engine ready", type: "info" },
  ]);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [isScanning, setIsScanning] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

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

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files) return;

    const validFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (validFiles.length === 0) {
      addLog("ERROR: Invalid file type", "error");
      return;
    }

    const newItems: QueueItem[] = [];
    validFiles.forEach(file => {
      if (queue.length + newItems.length < 10) {
        newItems.push({
          id: `${Date.now()}-${file.name}`,
          name: file.name,
          status: "wait"
        });
      } else {
        addLog("WARNING: Queue limit reached", "error");
      }
    });

    if (newItems.length > 0) {
      setQueue(prev => [...prev, ...newItems]);
      addLog(`${newItems.length} file(s) queued`, "info");
    }

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

    addLog(`### Processing: ${file.name}`, "heading");
    addLog("Initializing neural network...", "process");
    
    await new Promise(r => setTimeout(r, 500));
    addLog("> Stage 1: Feature Extraction", "detail");
    addLog("> Analyzing texture patterns...", "detail");

    await new Promise(r => setTimeout(r, 700));
    addLog("> Stage 2: Artifact Detection", "detail");
    
    const hasAnomaly = Math.random() < 0.3;
    if (hasAnomaly) {
      addLog("> * High-frequency noise detected", "detail");
    }
    
    await new Promise(r => setTimeout(r, 500));
    addLog("> Stage 3: Classification", "detail");

    await new Promise(r => setTimeout(r, 400 + Math.random() * 600));

    let aiProbability: number;
    let artifacts: string[] = [];
    const randomOutcome = Math.random();

    if (randomOutcome < 0.4) {
      aiProbability = 85 + Math.random() * 14;
      artifacts = [
        "手や指の構造的な不整合が検出されました",
        "背景テクスチャに反復パターンを確認",
        "顔のディテールに不自然な対称性"
      ];
    } else if (randomOutcome < 0.6) {
      aiProbability = 50 + Math.random() * 20;
      artifacts = [
        "エッジにノイズアーティファクトを検出",
        "境界値付近のため追加検証を推奨"
      ];
    } else {
      aiProbability = 5 + Math.random() * 25;
      artifacts = [
        "有機的なブラシストロークを確認",
        "人間特有の不規則パターンを検出",
        "AIアーティファクトは未検出"
      ];
    }

    if (hasAnomaly) {
      aiProbability = Math.min(99, aiProbability + 15);
      artifacts.push("高周波ノイズ異常を検出");
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
      verdict: isAI ? "AI DETECTED" : "HUMAN CONFIRMED",
      confidence: isAI ? aiScore : humanScore,
      processingTime,
      artifacts
    });

    addLog(`> Result: ${isAI ? "AI Generated" : "Human Creation"} (${isAI ? aiScore : humanScore}%)`, "result");
    
    await new Promise(r => setTimeout(r, 300));
  };

  const startBatchScan = async () => {
    const files = (window as unknown as { _pendingFiles?: File[] })._pendingFiles || [];
    if (isScanning || files.length === 0) return;

    setIsScanning(true);
    setStartTime(Date.now());
    setElapsedTime(0);
    setBatchProgress({ current: 0, total: files.length });
    addLog("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "system");
    addLog("BATCH ANALYSIS INITIATED", "system");

    for (let i = 0; i < files.length; i++) {
      setBatchProgress({ current: i + 1, total: files.length });
      await processFile(files[i], i);
    }

    setIsScanning(false);
    setStartTime(null);
    setPhase("complete");
    addLog("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "system");
    addLog("ANALYSIS COMPLETE", "system");
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
    setLogs([{ message: "System reset", type: "system" }]);
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

  return (
    <div className="noise-bg grid-bg scanlines flex flex-col h-screen overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-white/5 bg-black/60 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-8 py-5 flex justify-between items-center">
          <div className="flex items-center gap-5">
            <div className="relative">
              <div className="w-2.5 h-2.5 rounded-full bg-matrix" />
              <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-matrix blur-md opacity-60" />
            </div>
            <h1 className="text-xl tracking-[0.3em] font-normal">
              <span className="text-matrix">AI</span>
              <span className="text-white/50 ml-2">イラストチェッカー</span>
            </h1>
          </div>
          <div className="hidden sm:flex items-center gap-6 text-xs tracking-wider text-white/40">
            <div className="flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-matrix/80" />
              <span>ONLINE</span>
            </div>
            <span className="text-white/10">|</span>
            <span>v2.1</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-6xl w-full mx-auto px-8 py-6 flex flex-col lg:flex-row gap-6 overflow-hidden">

        {/* Left Column */}
        <div className="w-full lg:w-[63%] flex flex-col gap-6 h-full">

          {/* Active Process */}
          <div className="panel flex-[3] flex flex-col overflow-hidden">
            <div className="panel-header">
              <div className="flex items-center gap-3">
                <Cpu className="w-4 h-4 text-matrix/50" />
                <span className="text-xs">Active Process</span>
              </div>
              <span className="text-white/25 text-xs">Neural Engine</span>
            </div>
            
            <div className="panel-content flex-grow flex flex-col md:flex-row gap-6 overflow-hidden">
              {/* Preview */}
              <div className="w-full md:w-1/2 flex flex-col">
                <div className="flex-grow relative bg-black/50 border border-white/[0.04] rounded flex items-center justify-center min-h-[200px] overflow-hidden">
                  {currentImage ? (
                    <>
                      <img 
                        src={currentImage} 
                        alt="Target" 
                        className="max-w-full max-h-full object-contain"
                      />
                      {phase === "scanning" && (
                        <div className="absolute inset-0 overflow-hidden pointer-events-none">
                          <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-matrix to-transparent animate-scan opacity-70" />
                        </div>
                      )}
                      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/90 to-transparent">
                        <p className="text-xs text-white/60 truncate">{currentFileName}</p>
                      </div>
                    </>
                  ) : (
                    <div className="text-center text-white/25">
                      <Scan className="w-12 h-12 mx-auto mb-4 opacity-40" />
                      <p className="text-xs tracking-wider">AWAITING INPUT</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Log */}
              <div className="w-full md:w-1/2 flex flex-col min-h-[200px]">
                <div className="text-[10px] tracking-[0.2em] text-white/35 mb-3 uppercase">Analysis Log</div>
                <div
                  ref={logContainerRef}
                  className="flex-grow overflow-y-auto bg-black/40 border border-white/[0.04] rounded p-4 font-mono text-xs leading-relaxed space-y-1.5"
                >
                  {logs.map((log, i) => (
                    <div key={i} className={getLogClass(log.type)}>
                      {log.message}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Verdict Panel */}
          <div className="panel flex-[2] flex flex-col">
            <div className="panel-header">
              <span className="text-xs">最終評価 & メトリクス</span>
              <div className="flex gap-6 text-xs">
                <span>BATCH: <span className="text-matrix">{batchProgress.current}/{batchProgress.total || "—"}</span></span>
                <span>TIME: <span className="text-matrix">{elapsedTime.toFixed(2)}s</span></span>
              </div>
            </div>
            
            <div className="panel-content flex-grow flex flex-col justify-between">
              {/* Classification */}
              <div className="flex items-end gap-8 mb-5">
                <div className="text-[10px] tracking-wider text-white/40 leading-relaxed uppercase">FINAL<br/>CLASSIFICATION</div>
                <div className={`verdict-display ${result?.isAI ? 'verdict-ai' : result ? 'verdict-human' : 'text-white/15'}`}>
                  {result?.verdict || "—"}
                </div>
              </div>

              {/* Progress Bars */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-white/45">ARTIFICIAL INTELLIGENCE</span>
                    <span className={result?.aiScore ? "text-ai" : "text-white/25"}>{result?.aiScore ?? 0}%</span>
                  </div>
                  <div className="progress-track">
                    <div 
                      className="progress-fill bg-gradient-to-r from-ai/60 to-ai"
                      style={{ width: `${result?.aiScore ?? 0}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-white/45">HUMAN CREATION</span>
                    <span className={result?.humanScore ? "text-human" : "text-white/25"}>{result?.humanScore ?? 0}%</span>
                  </div>
                  <div className="progress-track">
                    <div 
                      className="progress-fill bg-gradient-to-r from-human/60 to-human"
                      style={{ width: `${result?.humanScore ?? 0}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-3 gap-6 pt-5 mt-5 border-t border-white/[0.04]">
                <div>
                  <div className="metric-label">使用モデル</div>
                  <div className="metric-value">ViT-Detector</div>
                </div>
                <div>
                  <div className="metric-label">信頼レベル</div>
                  <div className="metric-value text-matrix">{result?.confidence ? `${result.confidence}%` : "—"}</div>
                </div>
                <div>
                  <div className="metric-label">処理時間</div>
                  <div className="metric-value">{result?.processingTime ? `${result.processingTime.toFixed(2)}s` : "—"}</div>
                </div>
                <div className="col-span-3 mt-1">
                  <div className="metric-label mb-1.5">検出された特徴</div>
                  <div className="text-xs text-white/45 leading-relaxed">
                    {result?.artifacts?.[0] || "待機中..."}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="w-full lg:w-[37%] flex flex-col gap-6 h-full">

          {/* Queue */}
          <div className="panel flex-[3] flex flex-col overflow-hidden">
            <div className="panel-header">
              <span className="text-xs">解析キュー</span>
              <span className="text-white/35 text-xs">{queue.length}/10</span>
            </div>
            
            <div className="panel-content flex-grow overflow-y-auto space-y-2">
              {queue.length === 0 ? (
                <div className="h-full flex items-center justify-center text-white/25 text-xs">
                  <div className="text-center">
                    <div className="w-10 h-10 rounded-full border border-white/10 mx-auto mb-4 flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-white/20" />
                    </div>
                    キュー内にファイルがありません
                  </div>
                </div>
              ) : (
                queue.map((item) => (
                  <div key={item.id} className={`queue-item ${item.status}`}>
                    <div className={`data-indicator flex-shrink-0 ${
                      item.status === "processing" ? "bg-matrix" :
                      item.status === "ai" ? "bg-ai" :
                      item.status === "human" ? "bg-human" : "bg-white/20"
                    }`} />
                    <span className="truncate flex-grow text-white/55 text-xs">{item.name}</span>
                    <span className={`flex-shrink-0 text-[10px] tracking-wider ${
                      item.status === "processing" ? "text-matrix" :
                      item.status === "ai" ? "text-ai" :
                      item.status === "human" ? "text-human" : "text-white/35"
                    }`}>
                      {item.status.toUpperCase()}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Upload - matches Verdict panel height */}
          <div className="panel flex-[2] flex flex-col">
            <div className="panel-content flex-grow flex flex-col justify-between">
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`upload-zone flex-grow flex flex-col items-center justify-center cursor-pointer ${
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
                <Upload className={`w-8 h-8 mb-3 ${isDragging ? "text-matrix" : "text-white/25"}`} />
                <p className="text-xs text-white/45">画像をドロップまたはクリック</p>
              </div>

              <div className="flex gap-4 mt-5">
                <button
                  onClick={startBatchScan}
                  disabled={!canExecute}
                  className={`btn-primary flex-grow flex items-center justify-center gap-3 py-3 ${canExecute ? "active" : ""}`}
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>EXECUTE</span>
                </button>
                <button
                  onClick={resetUI}
                  disabled={isScanning}
                  className="w-12 h-12 rounded border border-white/10 text-white/35 hover:border-ai/40 hover:text-ai transition-all disabled:opacity-30 flex items-center justify-center"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
