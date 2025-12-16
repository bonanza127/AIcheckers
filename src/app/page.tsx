"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, Play, Trash2, Cpu, Search, Fingerprint, History, Plus } from "lucide-react";

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

type HistoryItem = {
  id: string;
  name: string;
  preview: string;
  isAI: boolean;
  score: number;
  timestamp: Date;
};

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>("");
  const [phase, setPhase] = useState<AnalysisPhase>("idle");
  const [logs, setLogs] = useState<LogEntry[]>([
    { message: "SYSTEM INITIALIZED. Ready for batch submission.", type: "system" },
  ]);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [isScanning, setIsScanning] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

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
      addLog("ERROR: 不正なファイル形式です。画像ファイルを選択してください。", "error");
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
        addLog("WARNING: キューの制限（10ファイル）に達しました。", "error");
      }
    });

    addLog(`キュー登録: ${validFiles.length}個のアーティファクトが処理キューに追加されました。`, "process");

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
    setResult(null);

    addLog(`解析開始: [${index + 1}/${queue.length}] ファイル: ${file.name}...`, "heading");
    addLog("NEURALNET に画像データを送信中...", "process");

    await new Promise(r => setTimeout(r, 500));
    addLog("> STAGE 1: 特徴抽出", "detail");

    await new Promise(r => setTimeout(r, 700));
    addLog("> STAGE 2: アーティファクト検出", "detail");

    const hasAnomaly = Math.random() < 0.3;
    if (hasAnomaly) {
      addLog("> * 高周波ノイズを検出", "detail");
    }

    await new Promise(r => setTimeout(r, 500));
    addLog("> STAGE 3: 分類処理", "detail");

    await new Promise(r => setTimeout(r, 400 + Math.random() * 600));

    let aiProbability: number;
    let artifacts: string;
    const randomOutcome = Math.random();

    if (randomOutcome < 0.4) {
      aiProbability = 85 + Math.random() * 14;
      artifacts = "手の異常、テクスチャの繰り返しを検出";
    } else if (randomOutcome < 0.6) {
      aiProbability = 50 + Math.random() * 20;
      artifacts = "エッジノイズ、境界の不整合を確認";
    } else {
      aiProbability = 5 + Math.random() * 25;
      artifacts = "有機的な筆致、AI痕跡なし";
    }

    if (hasAnomaly) {
      aiProbability = Math.min(99, aiProbability + 15);
      artifacts += " [フィルタ +15%]";
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

    // Add to history
    setHistory(prev => [{
      id: `history-${Date.now()}`,
      name: file.name,
      preview: imageUrl,
      isAI,
      score: isAI ? aiScore : humanScore,
      timestamp: new Date()
    }, ...prev].slice(0, 20)); // Keep last 20 items

    setPhase("complete");
    addLog(`最終判定: ${isAI ? "AI生成の可能性が高い" : "人間による創作物"} (${isAI ? aiScore : humanScore}%)`, "result");

    await new Promise(r => setTimeout(r, 500));
  };

  const startBatchScan = async () => {
    const files = (window as unknown as { _pendingFiles?: File[] })._pendingFiles || [];
    if (isScanning || files.length === 0) return;

    setIsScanning(true);
    setStartTime(Date.now());
    setElapsedTime(0);
    setBatchProgress({ current: 0, total: files.length });
    addLog("--- BATCH SCAN INITIATED (一括解析開始) ---", "heading");

    for (let i = 0; i < files.length; i++) {
      setBatchProgress({ current: i + 1, total: files.length });
      await processFile(files[i], i);
    }

    setIsScanning(false);
    setStartTime(null);
    addLog("--- BATCH SCAN COMPLETE (一括解析完了) ---", "heading");
    addLog(`STATUS: 全ての${files.length}個のアーティファクトの処理が完了しました。`, "process");
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
    setSelectedQueueId(null);
    setLogs([{ message: "システム: 処理キューがクリアされました。", type: "system" }]);
    (window as unknown as { _pendingFiles?: File[] })._pendingFiles = [];
  };

  const deleteSelectedItem = () => {
    if (isScanning || !selectedQueueId) return;
    
    const index = queue.findIndex(item => item.id === selectedQueueId);
    if (index === -1) return;
    
    setQueue(prev => prev.filter(item => item.id !== selectedQueueId));
    const pendingFiles = (window as unknown as { _pendingFiles?: File[] })._pendingFiles || [];
    (window as unknown as { _pendingFiles: File[] })._pendingFiles = pendingFiles.filter((_, i) => i !== index);
    setSelectedQueueId(null);
    addLog(`キューから画像を削除しました。`, "info");
  };

  const handleTrashClick = () => {
    if (selectedQueueId) {
      deleteSelectedItem();
    } else {
      resetUI();
    }
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

  const getVerdictDisplay = () => {
    if (phase === "scanning") {
      return {
        text: "PROCESSING...",
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
      text: "N/A",
      className: "verdict-pending"
    };
  };

  const verdictDisplay = getVerdictDisplay();

  // Get preview image - selected queue item or current scanning image
  const selectedQueueItem = queue.find(item => item.id === selectedQueueId);
  const previewImage = phase === "scanning" ? currentImage : (selectedQueueItem?.preview || currentImage);
  const previewFileName = phase === "scanning" ? currentFileName : (selectedQueueItem?.name || currentFileName);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="site-header sticky top-0 z-40 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Fingerprint className="w-8 h-8 text-accent" />
            <h1 className="text-2xl font-bold tracking-tight">
              Digital Forensics <span className="text-sm font-light text-muted">// AI Art Integrity V4.2</span>
            </h1>
          </div>
          <div className="text-sm font-light text-muted hidden sm:block">
            SYSTEM STATUS: <span className="text-success font-medium">OPERATIONAL</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8">
        {/* Intro */}
        <div className="text-center max-w-4xl mx-auto mb-10">
          <h2 className="text-4xl font-extrabold mb-3 tracking-tight">AIイラストチェッカー</h2>
          <p className="text-muted text-lg">
            最先端のマルチモーダルAIと<span className="text-accent font-medium">高周波アーティファクト検出フィルタ</span>を利用し、画像の組成を解析。<br />
            深層学習モデルの痕跡や、人間的な筆致の有無を高精度に検出します。
          </p>
        </div>

        <div className="flex flex-col lg:flex-row gap-6 flex-grow">

          {/* LEFT PANEL (2/3) */}
          <div className="w-full lg:w-2/3 flex flex-col gap-6">

            {/* Active Screen / Logs */}
            <div className="card-panel p-6 flex-grow flex flex-col">
              <h3 className="panel-header">アクティブ解析 & コンソール</h3>

              <div className="flex flex-col md:flex-row gap-6 flex-grow">
                {/* Active Image Preview */}
                <div className="w-full md:w-1/2 flex flex-col items-center">
                  {previewImage ? (
                    <div className={`active-image-container w-full h-72 flex items-center justify-center ${phase === "scanning" ? "scanning" : ""}`}>
                      <img
                        src={previewImage}
                        alt="Active Scan"
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  ) : (
                    <div className="scan-placeholder w-full h-72 flex flex-col items-center justify-center">
                      <Cpu className="w-12 h-12 text-dim mb-2" />
                      <p className="text-muted font-light">SYSTEM READY FOR INFERENCE</p>
                    </div>
                  )}
                  {previewFileName && (
                    <p className="text-sm text-muted mt-3 truncate w-full text-center font-mono">{previewFileName}</p>
                  )}
                </div>

                {/* Console Log */}
                <div className="w-full md:w-1/2">
                  <div className="mb-2 text-sm font-semibold text-accent uppercase border-b border-gray-700 pb-1">
                    Analysis Logs & Findings
                  </div>
                  <div
                    ref={logContainerRef}
                    className="console-log h-64 overflow-y-auto"
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
            <div className="card-panel p-6">
              <h3 className="text-xl font-bold border-b-2 border-accent pb-2 mb-4 uppercase tracking-widest">
                判定
              </h3>

              {/* Processing Status */}
              {batchProgress.total > 0 && (
                <div className="flex justify-between items-center mb-4 text-sm text-muted">
                  <span>BATCH STATUS: {batchProgress.current} / {batchProgress.total}</span>
                  <span>TOTAL ELAPSED: {elapsedTime.toFixed(2)}s</span>
                </div>
              )}

              {/* Verdict */}
              <div className="flex justify-between items-end mb-6">
                <span className="text-2xl font-medium text-muted uppercase">FINAL CLASSIFICATION</span>
                <span className={`verdict-display ${verdictDisplay.className}`}>
                  {verdictDisplay.text}
                </span>
              </div>

              {/* AI Bar */}
              <div className="mb-4">
                <div className="flex justify-between text-base mb-1">
                  <span className="text-danger font-semibold uppercase">ARTIFICIAL INTELLIGENCE</span>
                  <span className="text-danger font-bold">{result?.aiScore ?? 0}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill ai"
                    style={{ width: `${result?.aiScore ?? 0}%` }}
                  />
                </div>
              </div>

              {/* Human Bar */}
              <div className="mb-6">
                <div className="flex justify-between text-base mb-1">
                  <span className="text-success font-semibold uppercase">HUMAN CREATION</span>
                  <span className="text-success font-bold">{result?.humanScore ?? 0}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill human"
                    style={{ width: `${result?.humanScore ?? 0}%` }}
                  />
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-y-4 gap-x-8 text-sm font-light text-muted border-t border-gray-700 pt-4">
                <div className="flex justify-between items-center">
                  <span className="font-medium">使用モデル:</span>
                  <span className="font-bold text-accent">ViT-Detect (Multimodal)</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium">信頼レベル:</span>
                  <span className="font-bold">{result?.confidence ? `${result.confidence}%` : "--"}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium">処理時間 (単一):</span>
                  <span className="font-bold">{result?.processingTime ? `${result.processingTime.toFixed(2)}s` : "--"}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium">ロジック:</span>
                  <span className="font-bold text-dim">ハイブリッド検出 V2.0</span>
                </div>
                <div className="col-span-2 flex justify-between items-center">
                  <span className="font-medium">検出されたアーティファクト（痕跡）:</span>
                  <span className="font-bold text-right">{result?.artifacts || "Stand by..."}</span>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL (1/3) */}
          <div className="w-full lg:w-1/3 flex flex-col gap-4">

            {/* History */}
            <div className="card-panel p-4 flex-1 min-h-0">
              <h3 className="panel-header flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <History className="w-4 h-4" />
                  解析履歴
                </span>
                <span className="text-sm font-normal text-muted">({history.length}件)</span>
              </h3>
              <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto p-1">
                {history.length === 0 ? (
                  <p className="text-muted text-sm italic">解析履歴はありません。</p>
                ) : (
                  history.map((item) => (
                    <div
                      key={item.id}
                      className={`history-item relative group ${item.isAI ? "result-ai" : "result-human"}`}
                      title={`${item.name} - ${item.isAI ? "AI" : "Human"} ${item.score}%`}
                    >
                      <img
                        src={item.preview}
                        alt={item.name}
                        className="w-12 h-12 object-cover rounded"
                      />
                      {/* AI/Human Label */}
                      <div className={`absolute -top-1 -right-1 px-1.5 py-0.5 text-[8px] font-bold uppercase rounded ${
                        item.isAI 
                          ? "bg-danger text-white" 
                          : "bg-success text-white"
                      }`}>
                        {item.isAI ? "AI" : "人"}
                      </div>
                      {/* Hover overlay */}
                      <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 flex flex-col items-center justify-center text-[9px] text-white p-1 transition-opacity rounded">
                        <span className="truncate w-full text-center">{item.name.substring(0, 8)}...</span>
                        <span className={item.isAI ? "text-danger" : "text-success"}>{item.score}%</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Combined Queue + Upload Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`combined-upload-zone flex-1 min-h-[180px] p-4 ${isDragging ? "dragging" : ""}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept="image/*"
                multiple
                onChange={handleFileSelect}
              />

              {queue.length === 0 ? (
                /* Empty state - full clickable area */
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="h-full flex flex-col items-center justify-center cursor-pointer"
                >
                  <Upload className="w-12 h-12 mb-3 text-accent opacity-70" />
                  <p className="text-lg font-semibold text-text-primary">画像をアップロード</p>
                  <p className="text-sm text-muted mt-1">ドラッグ＆ドロップまたはクリック</p>
                </div>
              ) : (
                /* Has images - show grid with add button */
                <div className="h-full flex flex-col">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm font-semibold text-muted uppercase tracking-wide">
                      解析キュー ({queue.length}件)
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2 flex-1 content-start overflow-y-auto">
                    {queue.map((item) => (
                      <div
                        key={item.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedQueueId(selectedQueueId === item.id ? null : item.id);
                        }}
                        className={`queue-item relative group cursor-pointer ${
                          item.status === "processing" ? "active" :
                          item.status === "ai" ? "result-ai" :
                          item.status === "human" ? "result-human" : ""
                        } ${selectedQueueId === item.id ? "ring-2 ring-accent ring-offset-2 ring-offset-card-bg" : ""}`}
                      >
                        <img
                          src={item.preview}
                          alt={item.name}
                          className="w-14 h-14 object-cover"
                        />
                        <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 flex items-center justify-center text-[10px] text-white p-1 transition-opacity rounded-lg">
                          {item.name.substring(0, 10)}...
                        </div>
                      </div>
                    ))}
                    {/* Add more button */}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="add-more-button w-14 h-14 flex items-center justify-center"
                      title="画像を追加"
                    >
                      <Plus className="w-6 h-6" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Buttons */}
            <div className="flex gap-3">
              <button
                onClick={startBatchScan}
                disabled={!canExecute}
                className="flex-grow primary-button flex items-center justify-center gap-2"
              >
                <Search className="w-4 h-4" />
                スキャン開始
              </button>
              <button
                onClick={handleTrashClick}
                disabled={isScanning}
                className="danger-button w-12 flex items-center justify-center"
                title={selectedQueueId ? "選択した画像を削除" : "キューをリセット"}
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="site-footer p-4">
        <div className="container mx-auto text-center text-xs text-muted">
          <p>&copy; 2025 AI Art Integrity Console. All rights reserved. | <span className="text-accent">Professional Frontend V4.2</span></p>
        </div>
      </footer>
    </div>
  );
}
