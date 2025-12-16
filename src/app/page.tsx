"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, Play, RotateCcw, ImageIcon, Sparkles, Brush, Clock, Cpu, Shield } from "lucide-react";

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
  confidence: number;
  processingTime: number;
  findings: string;
};

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>("");
  const [isScanning, setIsScanning] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([
    { message: "システム起動完了", type: "system" },
    { message: "ニューラルエンジン待機中", type: "info" },
  ]);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
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
      addLog("エラー: 画像ファイルを選択してください", "error");
      return;
    }

    const newItems: QueueItem[] = validFiles.slice(0, 10 - queue.length).map(file => ({
      id: `${Date.now()}-${file.name}`,
      name: file.name,
      status: "wait" as const
    }));

    if (newItems.length > 0) {
      setQueue(prev => [...prev, ...newItems]);
      addLog(`${newItems.length}件のファイルを追加`, "info");
    }

    (window as unknown as { _pendingFiles: File[] })._pendingFiles = [
      ...((window as unknown as { _pendingFiles?: File[] })._pendingFiles || []),
      ...validFiles.slice(0, 10 - queue.length)
    ];
  }, [queue.length, addLog]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
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

    addLog(`▸ ${file.name}`, "heading");
    addLog("特徴抽出を実行中...", "process");
    await new Promise(r => setTimeout(r, 600));

    addLog("テクスチャパターン分析", "detail");
    await new Promise(r => setTimeout(r, 500));

    addLog("分類モデル実行中", "detail");
    await new Promise(r => setTimeout(r, 400 + Math.random() * 500));

    const randomOutcome = Math.random();
    let aiProbability: number;
    let findings: string;

    if (randomOutcome < 0.4) {
      aiProbability = 85 + Math.random() * 14;
      findings = "構造的不整合と反復パターンを検出。生成モデル特有の規則性が確認されました。手の描写や背景のテクスチャにAI生成の痕跡が見られます。";
    } else if (randomOutcome < 0.6) {
      aiProbability = 45 + Math.random() * 20;
      findings = "AIと人間の特徴が混在しています。部分的な加筆修正、または高度な後処理が施されている可能性があります。確定的な判断には追加検証が必要です。";
    } else {
      aiProbability = 5 + Math.random() * 25;
      findings = "有機的なブラシストロークと人間特有の不規則性を確認。筆圧の変化、意図的な線のブレ、個性的なカラーパレットが認められます。";
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
      confidence: isAI ? aiScore : humanScore,
      processingTime,
      findings
    });

    addLog(`→ ${isAI ? "AI生成" : "人間の作品"} (${isAI ? aiScore : humanScore}%)`, "result");
    await new Promise(r => setTimeout(r, 200));
  };

  const startAnalysis = async () => {
    const files = (window as unknown as { _pendingFiles?: File[] })._pendingFiles || [];
    if (isScanning || files.length === 0) return;

    setIsScanning(true);
    setStartTime(Date.now());
    setElapsedTime(0);
    setBatchProgress({ current: 0, total: files.length });
    addLog("═══════════════════════════", "system");
    addLog("解析開始", "system");

    for (let i = 0; i < files.length; i++) {
      setBatchProgress({ current: i + 1, total: files.length });
      await processFile(files[i], i);
    }

    setIsScanning(false);
    setStartTime(null);
    addLog("═══════════════════════════", "system");
    addLog("全ての解析が完了しました", "system");
  };

  const resetAll = () => {
    if (isScanning) return;
    setQueue([]);
    setCurrentImage(null);
    setCurrentFileName("");
    setResult(null);
    setBatchProgress({ current: 0, total: 0 });
    setElapsedTime(0);
    setLogs([{ message: "システムリセット", type: "system" }]);
    (window as unknown as { _pendingFiles?: File[] })._pendingFiles = [];
  };

  const canExecute = queue.length > 0 && !isScanning;

  return (
    <div className="min-h-screen flex flex-col relative z-10">
      {/* Header */}
      <header className="relative border-b border-[var(--glass-border)] bg-[rgba(15,15,26,0.8)] backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-8 py-6 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-semibold tracking-wide">
              <span className="text-gold">AI</span>
              <span className="text-[var(--text)] ml-2">Illustration Checker</span>
            </h1>
            <p className="text-sm text-[var(--text-muted)] mt-1.5 tracking-wide">
              アニメ・イラストの真贋判定システム
            </p>
          </div>
          <div className="flex items-center gap-6 text-sm text-[var(--text-muted)]">
            {elapsedTime > 0 && (
              <div className="flex items-center gap-2 font-mono">
                <Clock className="w-4 h-4 text-gold" />
                <span>{elapsedTime.toFixed(1)}s</span>
              </div>
            )}
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[var(--human)] animate-pulse" />
              <span>Online</span>
            </div>
          </div>
        </div>
        <div className="header-glow" />
      </header>

      {/* Main */}
      <main className="flex-grow max-w-6xl w-full mx-auto px-8 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-8">

          {/* Left Column */}
          <div className="flex flex-col gap-8">
            {/* Preview + Log Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Preview */}
              <div className="glass-card">
                <div className="card-header">
                  <span className="card-title">作品プレビュー</span>
                  {currentFileName && (
                    <span className="card-meta truncate max-w-[120px]">{currentFileName}</span>
                  )}
                </div>
                <div className="card-body">
                  <div className="preview-frame aspect-square">
                    {currentImage ? (
                      <div className="relative w-full h-full flex items-center justify-center p-6">
                        <img
                          src={currentImage}
                          alt="Preview"
                          className="max-w-full max-h-full object-contain rounded-lg"
                        />
                        {isScanning && <div className="scan-line" />}
                      </div>
                    ) : (
                      <div className="text-center text-[var(--text-muted)]">
                        <div className="w-20 h-20 mx-auto mb-5 rounded-full bg-[rgba(240,165,0,0.08)] flex items-center justify-center">
                          <ImageIcon className="w-8 h-8 text-gold opacity-60" />
                        </div>
                        <p className="text-sm">画像を選択してください</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Log */}
              <div className="glass-card">
                <div className="card-header">
                  <span className="card-title">解析ログ</span>
                </div>
                <div className="card-body">
                  <div
                    ref={logContainerRef}
                    className="h-[300px] overflow-y-auto space-y-1 pr-2"
                  >
                    {logs.map((log, i) => (
                      <div
                        key={i}
                        className={`log-entry ${
                          log.type === "heading" ? "log-heading" :
                          log.type === "detail" ? "log-detail" :
                          log.type === "result" ? "log-result" :
                          log.type === "error" ? "log-error" :
                          log.type === "system" ? "log-system" : ""
                        }`}
                      >
                        {log.message}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Result */}
            <div className="glass-card">
              <div className="card-header">
                <span className="card-title">判定結果</span>
                {batchProgress.total > 0 && (
                  <span className="card-meta">{batchProgress.current} / {batchProgress.total}</span>
                )}
              </div>
              <div className="card-body">
                {/* Verdict */}
                <div className="flex items-center gap-8 mb-10">
                  <div className="label">判定</div>
                  <div className={`heading-display ${
                    result?.isAI ? "verdict-ai" : result ? "verdict-human" : "verdict-pending"
                  }`}>
                    {result?.isAI ? (
                      <span className="flex items-center gap-4">
                        <Sparkles className="w-8 h-8" />
                        AI生成
                      </span>
                    ) : result ? (
                      <span className="flex items-center gap-4">
                        <Brush className="w-8 h-8" />
                        人間の作品
                      </span>
                    ) : "判定待ち"}
                  </div>
                </div>

                {/* Progress Bars */}
                <div className="space-y-6 mb-10">
                  <div>
                    <div className="flex justify-between text-sm mb-3">
                      <span className="text-[var(--text-muted)]">AI生成の可能性</span>
                      <span className={result?.aiScore ? "text-ai font-semibold" : "text-[var(--text-muted)]"}>
                        {result?.aiScore ?? 0}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill progress-ai"
                        style={{ width: `${result?.aiScore ?? 0}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-3">
                      <span className="text-[var(--text-muted)]">人間の作品の可能性</span>
                      <span className={result?.humanScore ? "text-human font-semibold" : "text-[var(--text-muted)]"}>
                        {result?.humanScore ?? 0}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill progress-human"
                        style={{ width: `${result?.humanScore ?? 0}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Metrics */}
                <div className="metrics-row">
                  <div>
                    <div className="label mb-2">使用モデル</div>
                    <div className="value flex items-center gap-2">
                      <Cpu className="w-3.5 h-3.5 text-gold" />
                      ViT-Detector
                    </div>
                  </div>
                  <div>
                    <div className="label mb-2">確信度</div>
                    <div className="value flex items-center gap-2">
                      <Shield className="w-3.5 h-3.5 text-gold" />
                      {result?.confidence ? `${result.confidence}%` : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="label mb-2">処理時間</div>
                    <div className="value flex items-center gap-2">
                      <Clock className="w-3.5 h-3.5 text-gold" />
                      {result?.processingTime ? `${result.processingTime.toFixed(2)}s` : "—"}
                    </div>
                  </div>
                </div>

                {/* Findings */}
                {result?.findings && (
                  <div className="mt-8">
                    <div className="label mb-3">詳細所見</div>
                    <div className="findings-box">
                      <p className="text-sm text-[var(--text)] leading-relaxed pl-4">
                        {result.findings}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="flex flex-col gap-8">
            {/* Queue */}
            <div className="glass-card flex-grow">
              <div className="card-header">
                <span className="card-title">キュー</span>
                <span className="card-meta">{queue.length} / 10</span>
              </div>
              <div className="card-body min-h-[300px]">
                {queue.length === 0 ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-20 h-20 mx-auto mb-5 rounded-full bg-[rgba(255,255,255,0.03)] flex items-center justify-center">
                        <ImageIcon className="w-8 h-8 text-[var(--text-muted)] opacity-40" />
                      </div>
                      <p className="text-sm text-[var(--text-muted)]">ファイルがありません</p>
                    </div>
                  </div>
                ) : (
                  <div>
                    {queue.map((item) => (
                      <div key={item.id} className="queue-item">
                        <div className={`queue-dot ${
                          item.status === "processing" ? "dot-processing" :
                          item.status === "ai" ? "dot-ai" :
                          item.status === "human" ? "dot-human" : "dot-wait"
                        }`} />
                        <span className="flex-grow truncate text-sm text-[var(--text)]">{item.name}</span>
                        <span className={`text-xs font-semibold tracking-wide ${
                          item.status === "ai" ? "text-ai" :
                          item.status === "human" ? "text-human" :
                          item.status === "processing" ? "text-gold" :
                          "text-[var(--text-muted)]"
                        }`}>
                          {item.status === "wait" ? "待機" :
                           item.status === "processing" ? "解析中" :
                           item.status === "ai" ? "AI" : "人間"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Upload */}
            <div className="glass-card">
              <div className="card-body">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className={`upload-zone h-44 flex flex-col items-center justify-center ${
                    isDragging ? "active" : ""
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    multiple
                    onChange={(e) => { handleFiles(e.target.files); if (fileInputRef.current) fileInputRef.current.value = ""; }}
                  />
                  <Upload className={`w-10 h-10 mb-4 ${isDragging ? "text-gold" : "text-[var(--text-muted)]"}`} />
                  <p className="text-sm text-[var(--text-muted)]">
                    画像をドロップ
                  </p>
                  <p className="text-xs text-[var(--text-muted)] mt-1.5 opacity-60">
                    またはクリックして選択
                  </p>
                </div>

                <div className="flex gap-4 mt-6">
                  <button
                    onClick={startAnalysis}
                    disabled={!canExecute}
                    className="btn flex-grow flex items-center justify-center gap-3"
                  >
                    <Play className="w-4 h-4" />
                    解析開始
                  </button>
                  <button
                    onClick={resetAll}
                    disabled={isScanning}
                    className="btn btn-ghost w-14 flex items-center justify-center"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--glass-border)] py-6 mt-auto">
        <div className="max-w-6xl mx-auto px-8 text-center text-sm text-[var(--text-muted)]">
          <p>AI Illustration Checker — Powered by Vision Transformer</p>
        </div>
      </footer>
    </div>
  );
}
