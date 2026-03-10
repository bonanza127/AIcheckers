"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Upload, Play, Trash2, Cpu, Search, History, Plus, Eye, EyeOff } from "lucide-react";
import VipModal from "@/components/VipModal";
import HamburgerMenu from "@/components/HamburgerMenu";
import getApiUrl from "@/lib/api";

type AnalysisPhase = "idle" | "scanning" | "complete";

type LogEntry = {
  message: string;
  type: "system" | "info" | "process" | "result" | "error" | "heading" | "detail";
};

type QueueItem = {
  id: string;
  name: string;
  preview: string;
  status: "wait" | "processing" | "ai" | "human" | "unknown";
  file?: File;  // 順序を保証するためqueueにFileを保持
};

type DetectionResult = {
  isAI: boolean;
  aiScore: number;
  humanScore: number;
  verdict: string;
  confidence: number;
  processingTime: number;
  artifacts: string;
  attentionMap?: string; // Base64エンコードされたヒートマップ
};

type HistoryItem = {
  id: string;
  name: string;
  preview: string;
  isAI: boolean;
  score: number;
  aiScore: number;
  attentionMap?: string;
  artifacts?: string; // detected_traces を保存
  timestamp: Date;
};

type AuthUser = {
  name: string;
  email: string;
  token: string;
  isVip: boolean;
  isAdmin?: boolean;
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
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true); // デフォルトでヒートマップ表示
  const [backendStatus, setBackendStatus] = useState<"online" | "degraded" | "offline" | null>(null);
  const [selectedModel, setSelectedModel] = useState<"anixplore" | "legekka" | "dinov3">("dinov3");
  const [urlInput, setUrlInput] = useState("");
  const [isLoadingUrl, setIsLoadingUrl] = useState(false);
  const [rateLimitRemaining, setRateLimitRemaining] = useState<number | null>(null);
  const [timeUntilReset, setTimeUntilReset] = useState("--:--:--");
  const [isVipModalOpen, setIsVipModalOpen] = useState(false);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true); // 認証状態確認中

  const fileInputRef = useRef<HTMLInputElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const logQueueRef = useRef<LogEntry[]>([]);
  const logRafRef = useRef<number | null>(null);
  const PERF_LOG = process.env.NEXT_PUBLIC_PERF_LOG === "true" || process.env.NODE_ENV === "development";

  // OAuthコールバック処理 & ログイン状態復元 & VIP決済結果処理
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const authStatus = params.get("auth");
    const vipStatus = params.get("vip");
    const magicToken = params.get("magic_token");
    const magicError = params.get("error");

    // マジックリンクエラー処理
    if (magicError === "magic_link_expired") {
      setLogs(prev => [...prev, {
        type: "error" as const,
        message: "マジックリンクの有効期限が切れています。"
      }]);
      window.history.replaceState({}, "", window.location.pathname);
      setIsAuthLoading(false);
      return;
    } else if (magicError === "invalid_magic_link") {
      setLogs(prev => [...prev, {
        type: "error" as const,
        message: "無効なマジックリンクです。"
      }]);
      window.history.replaceState({}, "", window.location.pathname);
      setIsAuthLoading(false);
      return;
    }

    // マジックリンクログイン処理
    if (magicToken) {
      const name = params.get("name") || "Developer";
      const email = params.get("email") || "";
      const isVip = params.get("is_vip") === "true";
      const isAdmin = params.get("is_admin") === "true";

      setAuthUser({ name, email, token: magicToken, isVip, isAdmin });
      localStorage.setItem("auth_token", magicToken);
      setLogs(prev => [...prev, {
        type: "system" as const,
        message: `🔑 Developer access granted. Welcome, ${name}!`
      }]);
      window.history.replaceState({}, "", window.location.pathname);
      setIsAuthLoading(false);
      return;
    }

    // VIP決済成功/キャンセル処理
    if (vipStatus === "success") {
      // VIP決済成功 → ユーザー情報を再取得してVIPステータスを更新
      const savedToken = localStorage.getItem("auth_token");
      if (savedToken) {
        const apiUrl = getApiUrl();
        fetch(`${apiUrl}/auth/me`, {
          headers: { Authorization: `Bearer ${savedToken}` }
        })
          .then(res => res.ok ? res.json() : Promise.reject())
          .then(data => {
            setAuthUser({
              name: data.name,
              email: data.email,
              token: savedToken,
              isVip: data.is_vip,
              isAdmin: data.is_admin
            });
            // 成功ログを追加
            setLogs(prev => [...prev, {
              type: "system" as const,
              message: "🎉 VIP登録が完了しました！ありがとうございます。"
            }]);
          })
          .catch(() => {
            console.error("Failed to fetch user info after VIP payment");
          })
          .finally(() => setIsAuthLoading(false));
      } else {
        // トークンがない場合もローディングを解除
        setIsAuthLoading(false);
      }
      // URLパラメータをクリア
      window.history.replaceState({}, "", window.location.pathname);
      return;
    } else if (vipStatus === "cancelled" || vipStatus === "error") {
      // 決済キャンセルまたはエラー
      console.log("VIP payment cancelled or failed:", params.get("message"));
      window.history.replaceState({}, "", window.location.pathname);
      setIsAuthLoading(false);
      return;
    }

    if (authStatus === "success") {
      const token = params.get("token");
      const name = params.get("name");
      const email = params.get("email");
      const isVip = params.get("is_vip") === "true";
      const isAdmin = params.get("is_admin") === "true";

      if (token && name && email) {
        setAuthUser({ name, email, token, isVip, isAdmin });
        localStorage.setItem("auth_token", token);
        setIsVipModalOpen(true); // VIPモーダルを開いて決済へ
      }
      // URLパラメータをクリア
      window.history.replaceState({}, "", window.location.pathname);
      setIsAuthLoading(false);
    } else if (authStatus === "error") {
      console.error("OAuth error:", params.get("message"));
      window.history.replaceState({}, "", window.location.pathname);
      setIsAuthLoading(false);
    } else {
      // ページリロード時: localStorageからトークン復元
      const savedToken = localStorage.getItem("auth_token");
      if (savedToken) {
        const apiUrl = getApiUrl();
        fetch(`${apiUrl}/auth/me`, {
          headers: { Authorization: `Bearer ${savedToken}` }
        })
          .then(res => res.ok ? res.json() : Promise.reject())
          .then(data => {
            setAuthUser({
              name: data.name,
              email: data.email,
              token: savedToken,
              isVip: data.is_vip,
              isAdmin: data.is_admin
            });
          })
          .catch(() => {
            // トークン無効 → 削除
            localStorage.removeItem("auth_token");
          })
          .finally(() => setIsAuthLoading(false));
      } else {
        setIsAuthLoading(false);
      }
    }
  }, []);

  // ヘルスチェックポーリング（30秒ごと）
  useEffect(() => {
    const apiUrl = getApiUrl();
    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) { setBackendStatus("offline"); return; }
        const data = await res.json();
        const s = data.status;
        setBackendStatus(s === "online" ? "online" : s === "degraded" ? "degraded" : "offline");
      } catch {
        setBackendStatus("offline");
      }
    };
    checkHealth();
    const id = setInterval(checkHealth, 30_000);
    return () => clearInterval(id);
  }, []);

  // リセットまでのカウントダウン
  useEffect(() => {
    const updateCountdown = () => {
      const now = new Date();
      const midnight = new Date(now);
      midnight.setHours(24, 0, 0, 0);
      const diff = midnight.getTime() - now.getTime();

      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);

      setTimeUntilReset(
        `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
      );
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);
    return () => clearInterval(interval);
  }, []);

  const addLog = useCallback((message: string, type: LogEntry["type"] = "info") => {
    logQueueRef.current.push({ message, type });
    if (logRafRef.current !== null) return;
    logRafRef.current = window.requestAnimationFrame(() => {
      const queued = logQueueRef.current.splice(0);
      logRafRef.current = null;
      if (queued.length === 0) return;
      setLogs(prev => [...prev, ...queued].slice(-500));
    });
  }, []);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    if (!PERF_LOG || typeof PerformanceObserver === "undefined") return;

    let observer: PerformanceObserver | null = null;
    try {
      observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          const dur = entry.duration || 0;
          if (dur >= 200) {
            addLog(`> [PERF] longtask ${dur.toFixed(1)}ms`, "detail");
          }
        });
      });
      observer.observe({ entryTypes: ["longtask"] });
    } catch {
      // best-effort
    }

    return () => {
      if (observer) observer.disconnect();
    };
  }, [PERF_LOG, addLog]);

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
      const reader = new FileReader();
      reader.onload = (e) => {
        const newItem: QueueItem = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}-${file.name}`,
          name: file.name,
          preview: e.target?.result as string,
          status: "wait",
          file  // Fileオブジェクトを保持（順序保証用）
        };
        setQueue(prev => [...prev, newItem]);
      };
      reader.readAsDataURL(file);
    });

    addLog(`キュー登録: ${validFiles.length}個のアーティファクトが処理キューに追加されました。`, "process");
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

  const handleUrlSubmit = useCallback(async () => {
    if (!urlInput.trim()) return;

    setIsLoadingUrl(true);
    addLog(`URL解析開始: ${urlInput}`, "process");

    try {
      const apiUrl = getApiUrl();
      const headers: HeadersInit = { "Content-Type": "application/json" };
      if (authUser?.token) {
        headers["Authorization"] = `Bearer ${authUser.token}`;
      }
      const response = await fetch(`${apiUrl}/analyze-url`, {
        method: "POST",
        headers,
        body: JSON.stringify({ url: urlInput, model: selectedModel })
      });

      // レート制限ヘッダーを読み取り
      const remaining = response.headers.get("X-RateLimit-Remaining");
      if (remaining !== null) {
        setRateLimitRemaining(parseInt(remaining, 10));
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "URL解析に失敗しました");
      }

      const data = await response.json();

      // 結果を表示
      const detectionResult: DetectionResult = {
        isAI: data.is_ai,
        aiScore: data.ai_score,
        humanScore: data.human_score,
        verdict: data.verdict,
        confidence: data.confidence,
        processingTime: data.processing_time,
        artifacts: data.detected_traces || "分析完了",
        attentionMap: data.attention_map
      };

      setResult(detectionResult);
      setPhase("complete");
      setCurrentFileName(data.filename || "URL Image");

      // 画像プレビューを設定（attention_mapがあればそれを使用）
      if (data.attention_map) {
        setCurrentImage(`data:image/png;base64,${data.attention_map}`);
      }

      // ログ出力
      addLog("═══════════════════════════════════════", "heading");
      addLog(`[ANALYZED] URL: ${urlInput}`, "heading");
      addLog("═══════════════════════════════════════", "heading");
      addLog(`[MODEL] ${data.model_used.toUpperCase()}`, "info");
      addLog(`[VERDICT] ${data.verdict}`, detectionResult.isAI ? "error" : "result");
      addLog(`[SCORE] AI: ${data.ai_score.toFixed(1)}% | Human: ${data.human_score.toFixed(1)}%`, "detail");

      if (data.forensic_logs) {
        data.forensic_logs.forEach((log: string) => addLog(`  ${log}`, "detail"));
      }

      // 履歴に追加（メモリのみ、元画像保持）
      const historyItem: HistoryItem = {
        id: `url-${Date.now()}`,
        name: data.filename || "URL Image",
        preview: data.attention_map ? `data:image/png;base64,${data.attention_map}` : "",
        isAI: data.is_ai,
        score: data.ai_score,
        aiScore: data.ai_score,
        attentionMap: data.attention_map,
        artifacts: data.detected_traces,
        timestamp: new Date()
      };
      setHistory(prev => [historyItem, ...prev].slice(0, 100));

      setUrlInput("");
      addLog(`URL解析完了: ${data.processing_time.toFixed(3)}秒`, "result");

    } catch (error) {
      addLog(`ERROR: ${error instanceof Error ? error.message : "URL解析に失敗しました"}`, "error");
    } finally {
      setIsLoadingUrl(false);
    }
  }, [urlInput, selectedModel, addLog]);

  const processFile = async (file: File, queueItemId: string, displayIndex: number, total: number, existingPreview?: string) => {
    const fileStartTime = Date.now();
    const perfStart = performance.now();
    const mark = (label: string) => {
      console.log(`[PERF] ${label}: ${(performance.now() - perfStart).toFixed(1)}ms`);
    };
    let fetchStart = 0;
    let fetchEnd = 0;
    let jsonStart = 0;
    let jsonEnd = 0;

    console.log(`[PERF] processFile START: ${file.name}`);

    setQueue(prev => prev.map(item =>
      item.id === queueItemId ? { ...item, status: "processing" as const } : item
    ));
    mark("queue status updated");

    // キュー追加時に作成済みのプレビューを再利用（FileReaderの二重読み込みを回避）
    const imageUrl = existingPreview || URL.createObjectURL(file);
    mark("imageUrl set");

    // 処理開始時にphaseをscanningに設定（resultは前の結果を維持し、一瞬表示される）
    setPhase("scanning");
    setCurrentImage(imageUrl);
    setCurrentFileName(file.name);
    mark("state updates (phase/currentImage/fileName)");

    // ファイルサイズを取得
    const fileSizeKB = (file.size / 1024).toFixed(1);

    addLog(`解析開始: [${displayIndex}/${total}] ファイル: ${file.name}...`, "heading");
    addLog(`画像データ読み込み完了 (${fileSizeKB}KB)`, "process");

    // 演出用のログを段階的に表示
    const analysisLogs = [
      { delay: 400, message: "前処理: リサイズ → 224×224px", type: "detail" as const },
      { delay: 800, message: "Vision Transformer エンコーディング開始...", type: "process" as const },
      { delay: 1200, message: "パッチ分析: 196パッチを解析中...", type: "detail" as const },
      { delay: 1800, message: "高周波アーティファクト検出...", type: "detail" as const },
      { delay: 2400, message: "テクスチャパターン分析...", type: "detail" as const },
      { delay: 3000, message: "アテンションマップ生成完了", type: "process" as const },
    ];

    // 演出ログを非同期で追加
    analysisLogs.forEach(({ delay, message, type }) => {
      setTimeout(() => addLog(`> ${message}`, type), delay);
    });

    // API呼び出しと演出時間を並行実行（0秒: デバッグ用）
    const minScanTime = 0;
    const scanDelayPromise = new Promise(r => setTimeout(r, minScanTime));

    // API呼び出し
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", selectedModel);

    let aiScore: number;
    let humanScore: number;
    let isAI: boolean;
    let processingTime: number;
    let artifacts: string;
    let attentionMap: string | undefined;
    let rateLimitError = false;
    let backendVerdict: string | undefined;

    try {
      const apiUrl = getApiUrl();
      const headers: HeadersInit = {};
      if (authUser?.token) {
        headers["Authorization"] = `Bearer ${authUser.token}`;
      }
      mark("before fetch");
      fetchStart = performance.now();
      const fetchPromise = fetch(`${apiUrl}/analyze`, {
        method: "POST",
        body: formData,
        headers,
      }).then(resp => {
        fetchEnd = performance.now();
        return resp;
      });
      const [response] = await Promise.all([
        fetchPromise,
        scanDelayPromise
      ]);
      if (PERF_LOG) {
        const clen = response.headers.get("content-length");
        const cenc = response.headers.get("content-encoding");
        if (clen) {
          addLog(`> [PERF] response content-length=${clen} bytes`, "detail");
        }
        if (cenc) {
          addLog(`> [PERF] response content-encoding=${cenc}`, "detail");
        }
      }
      if (PERF_LOG) {
        const entries = performance.getEntriesByName(`${apiUrl}/analyze`).filter(e => (e as PerformanceResourceTiming).initiatorType === "fetch");
        const lastEntry = entries[entries.length - 1] as PerformanceResourceTiming | undefined;
        if (lastEntry) {
          const ttfb = lastEntry.responseStart - lastEntry.startTime;
          const download = lastEntry.responseEnd - lastEntry.responseStart;
          const dns = lastEntry.domainLookupEnd - lastEntry.domainLookupStart;
          const connect = lastEntry.connectEnd - lastEntry.connectStart;
          const tls = lastEntry.secureConnectionStart > 0 ? (lastEntry.connectEnd - lastEntry.secureConnectionStart) : 0;
          addLog(`> [PERF] net ttfb=${ttfb.toFixed(1)}ms download=${download.toFixed(1)}ms dns=${dns.toFixed(1)}ms connect=${connect.toFixed(1)}ms tls=${tls.toFixed(1)}ms`, "detail");
        }
      }
      mark("fetch response received");

      // レート制限ヘッダーを読み取り
      const remaining = response.headers.get("X-RateLimit-Remaining");
      if (remaining !== null) {
        setRateLimitRemaining(parseInt(remaining, 10));
      }

      if (!response.ok) {
        if (response.status === 429) {
          const errorData = await response.json().catch(() => ({ detail: "レート制限に達しました" }));
          throw new Error(`RATE_LIMITED:${errorData.detail || "レート制限に達しました"}`);
        }
        throw new Error(`API error: ${response.status}`);
      }

      jsonStart = performance.now();
      const data = await response.json();
      jsonEnd = performance.now();
      mark("JSON parsed");

      aiScore = Math.round(data.ai_score);
      humanScore = Math.round(data.human_score);
      isAI = aiScore >= 80; // 80%以上でAI判定
      processingTime = data.processing_time;
      attentionMap = data.attention_map; // Attention Mapを取得
      backendVerdict = data.verdict; // バックエンドのverdictを取得

      // detected_tracesがあればそれを優先、なければforensic_logsから生成
      const detectedTraces: string = data.detected_traces || "";
      const forensicLogs: string[] = data.forensic_logs || [];

      if (forensicLogs.length > 0) {
        // バックエンドからのforensic_logsを使用
        forensicLogs.forEach((log: string) => {
          const logType = log.includes("判定") ? (aiScore >= 80 ? "error" : aiScore >= 50 ? "info" : "process") : "detail";
          addLog(`> ${log}`, logType);
        });
      }

      // 検出された痕跡の設定（優先順位: detected_traces > forensic_logs > フォールバック）
      if (detectedTraces) {
        artifacts = detectedTraces;
      } else if (forensicLogs.length > 0) {
        const traces = forensicLogs.filter(l => !l.startsWith("判定")).slice(0, 2);
        artifacts = traces.length > 0 ? traces.join(" / ") : forensicLogs[0];
      } else {
        // フォールバック: 従来のロジック
        if (aiScore >= 80) {
          addLog("> 検出: 均一すぎるテクスチャパターン", "detail");
          addLog("> 検出: 不自然なエッジ処理の痕跡", "detail");
          addLog("> 警告: AI生成の特徴が顕著", "error");
          artifacts = "均一テクスチャ、不自然なエッジ処理";
        } else if (aiScore >= 50) {
          addLog("> 検出: AI/人間の特徴が混在", "detail");
          addLog("> 警告: 判定が困難な領域", "info");
          addLog("> 推奨: 追加の検証が必要", "info");
          artifacts = "特徴混在 - 追加検証を推奨";
        } else {
          addLog("> 検出: 有機的な筆致の揺らぎ", "detail");
          addLog("> 検出: 自然なテクスチャ分布", "detail");
          addLog("> 確認: AI生成の痕跡なし", "process");
          artifacts = "有機的筆致、自然なテクスチャ";
        }
      }

      addLog("> Attention Map生成完了", "process");
      addLog("> 推論完了", "process");
      if (PERF_LOG) {
        const totalMs = performance.now() - perfStart;
        const fetchMs = fetchEnd > 0 ? (fetchEnd - fetchStart) : 0;
        const jsonMs = jsonEnd > 0 && jsonStart > 0 ? (jsonEnd - jsonStart) : 0;
        const delayMs = Math.max(0, totalMs - fetchMs - jsonMs);
        const perfMsg = `client_total=${totalMs.toFixed(1)}ms fetch=${fetchMs.toFixed(1)}ms json=${jsonMs.toFixed(1)}ms delay=${delayMs.toFixed(1)}ms backend=${(processingTime * 1000).toFixed(1)}ms`;
        addLog(`> [PERF] ${perfMsg}`, "detail");
        try {
          fetch(`${apiUrl}/client-perf`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ts: Date.now(),
              file: file.name,
              client_total_ms: totalMs,
              fetch_ms: fetchMs,
              json_ms: jsonMs,
              delay_ms: delayMs,
              backend_ms: processingTime * 1000,
            }),
          });
        } catch {
          // best-effort
        }
      }

    } catch (error) {
      await scanDelayPromise; // エラー時も演出時間を確保
      const errorMessage = error instanceof Error ? error.message : String(error);

      if (errorMessage.startsWith("RATE_LIMITED:")) {
        // レート制限エラー
        const detail = errorMessage.replace("RATE_LIMITED:", "");
        addLog(`ERROR: ${detail}`, "error");
        rateLimitError = true;
        aiScore = 0;
        humanScore = 0;
        isAI = false;
        processingTime = (Date.now() - fileStartTime) / 1000;
        artifacts = detail;
        attentionMap = undefined;
      } else {
        // その他のエラー
        addLog(`ERROR: API接続失敗 - ${error}`, "error");
        aiScore = 50;
        humanScore = 50;
        isAI = false;
        processingTime = (Date.now() - fileStartTime) / 1000;
        artifacts = "API接続エラー";
        attentionMap = undefined;
      }
      if (PERF_LOG) {
        const totalMs = performance.now() - perfStart;
        const fetchMs = fetchEnd > 0 ? (fetchEnd - fetchStart) : 0;
        const jsonMs = jsonEnd > 0 && jsonStart > 0 ? (jsonEnd - jsonStart) : 0;
        const delayMs = Math.max(0, totalMs - fetchMs - jsonMs);
        const perfMsg = `client_total=${totalMs.toFixed(1)}ms fetch=${fetchMs.toFixed(1)}ms json=${jsonMs.toFixed(1)}ms delay=${delayMs.toFixed(1)}ms (error)`;
        addLog(`> [PERF] ${perfMsg}`, "detail");
        try {
          const apiUrl = getApiUrl();
          fetch(`${apiUrl}/client-perf`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ts: Date.now(),
              file: file.name,
              client_total_ms: totalMs,
              fetch_ms: fetchMs,
              json_ms: jsonMs,
              delay_ms: delayMs,
              error: true,
            }),
          });
        } catch {
          // best-effort
        }
      }
    }

    // キューのステータス更新（3段階 + レート制限）
    const queueStatus = rateLimitError ? "unknown" as const : aiScore >= 80 ? "ai" as const : aiScore >= 50 ? "unknown" as const : "human" as const;
    setQueue(prev => prev.map(item =>
      item.id === queueItemId ? { ...item, status: queueStatus } : item
    ));

    // verdict: バックエンドのverdictを優先、なければフロントエンドで計算
    // 5段階分類（低い順）: HUMAN CONFIRMED(青) < LOW SIMILARITY(緑) < MIDDLE CAUTION(黄) < HIGH ALERT(オレンジ) < AI DETECTED(赤)
    const verdict = rateLimitError ? "RATE LIMITED"
      : backendVerdict ? backendVerdict  // バックエンドのverdictを優先使用
        : aiScore >= 80 ? "AI DETECTED"
          : aiScore >= 60 ? "HIGH ALERT"
            : aiScore >= 40 ? "MIDDLE CAUTION"
              : aiScore >= 20 ? "LOW SIMILARITY"
                : "HUMAN CONFIRMED";

    mark("before setResult");
    setResult({
      isAI,
      aiScore,
      humanScore,
      verdict,
      confidence: isAI ? aiScore : humanScore,
      processingTime,
      artifacts,
      attentionMap
    });
    mark("after setResult");

    // Add to history（メモリのみ、元画像保持）
    setHistory(prev => [{
      id: `history-${Date.now()}`,
      name: file.name,
      preview: imageUrl,
      isAI,
      score: isAI ? aiScore : humanScore,
      aiScore,
      attentionMap,
      artifacts,
      timestamp: new Date()
    }, ...prev].slice(0, 100));

    // Remove from queue after scan complete
    setQueue(prev => prev.filter(item => item.id !== queueItemId));

    setPhase("complete");
    const logMessage = rateLimitError
      ? "エラー: レート制限に達しました"
      : aiScore >= 80
        ? `最終判定: AI生成の可能性が高い (${aiScore}%)`
        : aiScore >= 50
          ? `最終判定: 判定困難 (${aiScore}%)`
          : `最終判定: 人間による創作物 (${aiScore}%)`;
    addLog(logMessage, rateLimitError ? "error" : "result");

    console.log(`[PERF] processFile END: total ${(performance.now() - perfStart).toFixed(1)}ms`);

    // レート制限エラーを返す（バッチ中断用）
    return { rateLimitError };
  };

  const startBatchScan = async () => {
    // キュー内のwait状態のアイテムを処理（Fileオブジェクト付き）
    const waitingItems = queue.filter(item => item.status === "wait" && item.file);
    if (isScanning || waitingItems.length === 0) return;

    setIsScanning(true);
    setStartTime(Date.now());
    setElapsedTime(0);
    setBatchProgress({ current: 0, total: waitingItems.length });
    addLog("--- BATCH SCAN INITIATED (一括解析開始) ---", "heading");

    let completedCount = 0;
    for (let i = 0; i < waitingItems.length; i++) {
      const item = waitingItems[i];
      setBatchProgress({ current: i + 1, total: waitingItems.length });
      // queue内のFileとpreviewを使用（順序が保証される）
      const result = await processFile(item.file!, item.id, i + 1, waitingItems.length, item.preview);

      // レート制限に達したらバッチを中断
      if (result?.rateLimitError) {
        const remaining = waitingItems.length - i - 1;
        addLog(`--- BATCH SCAN ABORTED (レート制限により中断) ---`, "error");
        addLog(`STATUS: ${remaining}個のアーティファクトが未処理です。しばらく待ってから再試行してください。`, "error");
        break;
      }
      completedCount++;
    }

    setIsScanning(false);
    setStartTime(null);

    if (completedCount === waitingItems.length) {
      addLog("--- BATCH SCAN COMPLETE (一括解析完了) ---", "heading");
      addLog(`STATUS: 全ての${waitingItems.length}個のアーティファクトの処理が完了しました。`, "process");
    }
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
  };

  const deleteSelectedItem = () => {
    if (isScanning || !selectedQueueId) return;

    setQueue(prev => prev.filter(item => item.id !== selectedQueueId));
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

  // X（Twitter）に結果を共有（動的OGP対応）
  const shareToX = () => {
    if (!result) return;

    const verdictText = result.aiScore >= 80 ? "AI DETECTED"
      : result.aiScore >= 60 ? "HIGH ALERT"
        : result.aiScore >= 40 ? "MIDDLE CAUTION"
          : result.aiScore >= 20 ? "LOW SIMILARITY"
            : "HUMAN CONFIRMED";

    const verdictEmoji = result.aiScore >= 80 ? "🤖"
      : result.aiScore >= 60 ? "🟠"
        : result.aiScore >= 40 ? "🟡"
          : result.aiScore >= 20 ? "🟢"
            : "🔵";

    const text = `【AI判定結果】
${verdictEmoji} ${verdictText}
AI Possibility: ${result.aiScore.toFixed(1)}%

#AIイラスト判定 #aicheckers`;

    // 動的OGP付きのシェアURL（短縮パラメータ使用）
    const vParam = verdictText === "AI DETECTED" ? "ai"
      : verdictText === "HIGH ALERT" ? "ha"
        : verdictText === "MIDDLE CAUTION" ? "mc"
          : verdictText === "LOW SIMILARITY" ? "ls"
            : "h";
    // traceを短いコードに変換（URLを短縮するため）
    const getTraceCode = (artifacts: string): string => {
      if (artifacts.includes("均一テクスチャ") || artifacts.includes("不自然")) return "ai";
      if (artifacts.includes("特徴混在") || artifacts.includes("追加検証")) return "mx";
      if (artifacts.includes("有機的") || artifacts.includes("自然なテクスチャ")) return "hu";
      return "";
    };
    const traceCode = result.artifacts ? getTraceCode(result.artifacts) : "";
    const traceParam = traceCode ? `&tr=${traceCode}` : "";
    const shareUrl = `https://aicheckers.net/share?v=${vParam}&s=${Math.round(result.aiScore)}${traceParam}`;
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareUrl)}`;

    window.open(twitterUrl, "_blank", "width=550,height=420");
  };

  const canExecute = queue.length > 0 && !isScanning;

  // ファイル名を省略（拡張子を維持）
  const truncateFileName = (name: string, maxLength: number = 35) => {
    if (name.length <= maxLength) return name;
    const lastDot = name.lastIndexOf('.');
    const ext = lastDot > 0 ? name.slice(lastDot) : '';
    const nameWithoutExt = lastDot > 0 ? name.slice(0, lastDot) : name;
    const truncatedLength = maxLength - ext.length - 3; // 3 for "..."
    if (truncatedLength <= 0) return name.slice(0, maxLength - 3) + '...';
    return nameWithoutExt.slice(0, truncatedLength) + '...' + ext;
  };

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
    // スキャン中は常にPROCESSING...を表示
    if (phase === "scanning") {
      return { text: "PROCESSING...", className: "verdict-loading" };
    }
    // スキャン完了時のみ結果を表示
    if (result) {
      if (result.verdict === "RATE LIMITED") {
        return { text: result.verdict, className: "verdict-unknown" };
      }
      // 5段階の判定表示（aiScoreから計算）
      const score = result.aiScore;
      if (score >= 80) return { text: "AI DETECTED", className: "verdict-ai" };
      if (score >= 60) return { text: "HIGH ALERT", className: "verdict-high-alert" };
      if (score >= 40) return { text: "MIDDLE CAUTION", className: "verdict-middle-caution" };
      if (score >= 20) return { text: "LOW SIMILARITY", className: "verdict-low-risk" };
      return { text: "HUMAN CONFIRMED", className: "verdict-human" };
    }
    return { text: "N/A", className: "verdict-pending" };
  };

  const verdictDisplay = getVerdictDisplay();

  // Get preview image - selected queue item or current scanning image
  const selectedQueueItem = queue.find(item => item.id === selectedQueueId);
  const previewImage = phase === "scanning" ? currentImage : (selectedQueueItem?.preview || currentImage);
  const previewFileName = phase === "scanning" ? currentFileName : (selectedQueueItem?.name || currentFileName);

  // Handle history item click
  const handleHistoryClick = (item: HistoryItem) => {
    setSelectedHistoryId(item.id);
    setSelectedQueueId(null); // Clear queue selection
    setShowHeatmap(true); // Attention Mapをデフォルト表示
    setCurrentImage(item.preview);
    setCurrentFileName(item.name);
    // 保存されたartifactsを使用、なければフォールバック
    const fallbackArtifacts = item.aiScore >= 80 ? "均一テクスチャ、不自然なエッジ処理" : item.aiScore >= 50 ? "特徴混在 - 追加検証を推奨" : "有機的筆致、自然なテクスチャ";
    setResult({
      isAI: item.isAI,
      aiScore: item.aiScore,
      humanScore: 100 - item.aiScore,
      verdict: item.aiScore >= 80 ? "AI DETECTED"
        : item.aiScore >= 60 ? "HIGH ALERT"
          : item.aiScore >= 40 ? "MIDDLE CAUTION"
            : item.aiScore >= 20 ? "LOW SIMILARITY"
              : "HUMAN CONFIRMED",
      confidence: item.score,
      processingTime: 0,
      artifacts: item.artifacts || fallbackArtifacts,
      attentionMap: item.attentionMap
    });
    setPhase("complete");
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="site-header sticky top-0 z-40 p-4">
        <div className="container mx-auto flex justify-between items-center">
          {/* 左: メニュー + ロゴ */}
          <div className="flex items-center gap-0">
            <HamburgerMenu />
            <img src="/logo-transparent.png" alt="AI Checkers" className="w-14 h-14" />
            <h2 className="text-lg md:text-2xl font-bold tracking-tight whitespace-nowrap">
              AIチェッカー
              <span className="hidden md:inline text-sm font-light text-muted">　//　</span>
              <a href="/how-it-works" className="hidden md:inline text-sm font-medium text-muted hover:text-foreground hover:bg-white/5 px-2 py-1 rounded transition-all border border-transparent hover:border-gray-700">
                How it works?
              </a>
            </h2>
          </div>

          {/* 右: ステータス + VIP */}
          <div className="flex items-center gap-4 text-xs">
            {/* Server Status */}
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${
                backendStatus === null ? "bg-gray-500 animate-pulse"
                : backendStatus === "online" ? "bg-success"
                : backendStatus === "degraded" ? "bg-yellow-500"
                : "bg-danger"
              }`} />
              <span className="hidden md:inline text-muted">Server Status:</span>
              <span className={
                backendStatus === "online" ? "text-success"
                : backendStatus === "degraded" ? "text-yellow-500"
                : backendStatus === "offline" ? "text-danger"
                : "text-gray-500"
              }>
                {backendStatus === null ? "..." : backendStatus === "online" ? "Online" : backendStatus === "degraded" ? "No GPU" : "Offline"}
              </span>
            </div>

            {/* VIP - 控えめなブラックカード（ログイン時は紫の光） */}
            <button
              onClick={() => setIsVipModalOpen(true)}
              disabled={isAuthLoading}
              className={`group relative px-4 py-1.5 font-[family-name:var(--font-cinzel)] text-[10px] font-medium tracking-[0.2em] transition-all duration-500 bg-zinc-900/50 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] rounded-sm ${isAuthLoading
                ? "text-zinc-600 border border-zinc-800/50 cursor-wait"
                : authUser
                  ? "text-purple-400 border border-purple-500/40 shadow-[0_0_8px_rgba(168,85,247,0.15)] hover:border-purple-400/60 hover:shadow-[0_0_12px_rgba(168,85,247,0.25)]"
                  : "text-zinc-500 border border-zinc-700/50 hover:text-zinc-300 hover:border-zinc-600"
                }`}
            >
              {isAuthLoading ? "..." : "VIP"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8">
        {/* Intro */}
        <div className="text-center max-w-4xl mx-auto mb-10">
          <h1 className="text-4xl font-extrabold mb-3 tracking-tight">二次元に特化した、日本のためのAIイラストチェッカー</h1>
          <p className="text-muted text-lg">
            AIが生成したアニメ画像を学習し、ファインチューニングしたViTが生成画像の痕跡を解析。<br />
            人間的な筆致の有無を検出し、生成画像を<span className="text-accent font-bold">98.35%の精度</span><sup className="text-xs text-muted">*</sup>で判別します。
          </p>
          <p className="text-xs text-muted mt-2">
            * 学習済みモデルの精度。画像は一万枚で検証。
          </p>
        </div>

        <div className="flex flex-col lg:flex-row gap-6 flex-grow">

          {/* LEFT PANEL (2/3) */}
          <div className="w-full lg:w-2/3 flex flex-col gap-6">

            {/* Active Screen / Logs */}
            <div className="card-panel p-6 flex-grow flex flex-col">
              <h3 className="panel-header">リアルタイム解析 ＆ コンソール</h3>

              <div className="flex flex-col md:flex-row gap-6 flex-grow">
                {/* Active Image Preview */}
                <div className="w-full md:w-1/2 flex flex-col items-center">
                  {previewImage ? (
                    <div className="relative w-full">
                      <div className={`active-image-container w-full h-48 md:h-72 flex items-center justify-center ${phase === "scanning" ? "scanning" : ""}`}>
                        <img
                          src={showHeatmap && result?.attentionMap && phase === "complete" ? `data:image/png;base64,${result.attentionMap}` : previewImage}
                          alt={showHeatmap && phase === "complete" ? "Attention Heatmap" : "Active Scan"}
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                      {/* Heatmap Toggle Button - 常に表示 */}
                      {result?.attentionMap && (
                        <button
                          onClick={() => setShowHeatmap(!showHeatmap)}
                          className={`absolute top-2 right-2 p-2 rounded-lg transition-all ${showHeatmap
                            ? "bg-accent text-white"
                            : "bg-black/50 text-white hover:bg-accent/70"
                            }`}
                          title={showHeatmap ? "オリジナル画像を表示" : "Attention Mapを表示"}
                        >
                          {showHeatmap ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="scan-placeholder w-full h-48 md:h-72 flex flex-col items-center justify-center">
                      <Cpu className="w-12 h-12 text-dim mb-2" />
                      <p className="text-muted font-light">SYSTEM READY FOR INFERENCE</p>
                    </div>
                  )}
                  {previewFileName && (
                    <div className="flex items-center justify-center gap-2 mt-3">
                      <p className="text-sm text-muted font-mono" title={previewFileName}>{truncateFileName(previewFileName)}</p>
                      {showHeatmap && <span className="text-xs text-accent font-semibold">[ATTENTION MAP]</span>}
                    </div>
                  )}
                </div>

                {/* Console Log */}
                <div className="w-full md:w-1/2">
                  <div
                    ref={logContainerRef}
                    className="console-log h-48 md:h-72 overflow-y-auto"
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
            <div className="card-panel p-6 relative">
              <button
                onClick={shareToX}
                disabled={!result}
                className="absolute top-3 right-3 p-1.5 text-muted hover:text-white hover:drop-shadow-[0_0_6px_rgba(255,255,255,0.5)] disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                title="Xで結果を共有"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </button>
              <h3 className="text-xl font-bold border-b-2 border-accent pb-2 mb-4 uppercase tracking-widest">
                最終判定
              </h3>

              {/* Row 1: Batch Status + Model + Logic + Processing Time */}
              <div className="flex flex-wrap justify-between items-center mb-3 text-sm text-muted gap-2">
                <span>BATCH STATUS: {batchProgress.current || "-"} / {batchProgress.total || "-"}</span>
                <span>使用モデル: <span className="text-accent font-bold">Moonlight V1.3.6</span></span>
                <span>ロジック: <span className="text-dim font-bold">CLS with 手作り特徴</span></span>
                <span>PROCESSING TIME: <span className="font-bold">{elapsedTime.toFixed(2)}s</span></span>
              </div>

              {/* Row 2: Detected Artifacts */}
              <div className="flex items-start gap-2 text-sm text-muted mb-4">
                <span className="font-medium whitespace-nowrap">検出された痕跡:</span>
                <span className="font-bold text-text-primary">{result?.artifacts || "待機中..."}</span>
              </div>

              {/* AI Possibility Bar */}
              <div className="mb-6">
                <div className="flex justify-between text-base mb-1">
                  <span className="font-semibold uppercase text-danger">
                    AI POSSIBILITY
                  </span>
                  {result?.verdict === "RATE LIMITED" ? (
                    <span className="font-bold text-warning animate-pulse">
                      LIMIT
                    </span>
                  ) : (
                    <span className={`font-bold ${(result?.aiScore ?? 0) >= 80 ? "text-danger" :
                      (result?.aiScore ?? 0) >= 60 ? "text-high-alert" :
                        (result?.aiScore ?? 0) >= 40 ? "text-middle-caution" :
                          (result?.aiScore ?? 0) >= 20 ? "text-low-risk" : "text-human-confirmed"
                      }`}>
                      {result?.aiScore ?? 0}%
                    </span>
                  )}
                </div>
                <div className="progress-bar-bg">
                  <div
                    className={`progress-bar-fill ${result?.verdict === "RATE LIMITED" ? "rate-limited" :
                      (result?.aiScore ?? 0) >= 80 ? "ai" :
                        (result?.aiScore ?? 0) >= 60 ? "high-alert" :
                          (result?.aiScore ?? 0) >= 40 ? "middle-caution" :
                            (result?.aiScore ?? 0) >= 20 ? "low-risk" : "human"
                      }`}
                    style={{ width: result?.verdict === "RATE LIMITED" ? "100%" : `${result?.aiScore ?? 0}%` }}
                  />
                </div>
              </div>

              {/* Classification Result */}
              <div className="flex justify-between items-end border-t border-gray-700 pt-4">
                <span className="text-2xl font-medium text-muted uppercase">CLASSFICATION:</span>
                <span className={`verdict-display ${verdictDisplay.className}`}>
                  {verdictDisplay.text}
                </span>
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
                  履歴
                </span>
                <span className="text-sm font-normal text-muted">({history.length}件)</span>
              </h3>
              <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto p-1 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
                {history.length === 0 ? (
                  <p className="text-muted text-sm italic">解析履歴はありません。</p>
                ) : (
                  history.map((item) => {
                    // 5段階判定: AI(80+), H(60-79), M(40-59), L(20-39), 人(0-19)
                    const score = item.aiScore;
                    const resultClass = score >= 80 ? "result-ai" : score >= 60 ? "result-high" : score >= 40 ? "result-middle" : score >= 20 ? "result-low" : "result-human";
                    const labelClass = score >= 80 ? "bg-danger text-black" : score >= 60 ? "bg-high-alert text-black" : score >= 40 ? "bg-middle-caution text-black" : score >= 20 ? "bg-success text-black" : "bg-human-confirmed text-black";
                    const labelText = score >= 80 ? "AI" : score >= 60 ? "H" : score >= 40 ? "M" : score >= 20 ? "L" : "人";
                    const scoreClass = score >= 80 ? "text-danger" : score >= 60 ? "text-high-alert" : score >= 40 ? "text-middle-caution" : score >= 20 ? "text-low-risk" : "text-human-confirmed";

                    return (
                      <div
                        key={item.id}
                        onClick={() => handleHistoryClick(item)}
                        className={`history-item relative group cursor-pointer ${resultClass} ${selectedHistoryId === item.id ? "ring-2 ring-accent ring-offset-2 ring-offset-card-bg" : ""}`}
                        title={`${item.name} - ${item.aiScore}%`}
                      >
                        <img
                          src={item.preview}
                          alt={item.name}
                          className="w-12 h-12 object-cover rounded"
                        />
                        {/* AI/Human/Unknown Label */}
                        <div className={`absolute -top-1 -right-1 px-1.5 py-0.5 text-[8px] font-bold uppercase rounded ${labelClass}`}>
                          {labelText}
                        </div>
                        {/* Hover overlay */}
                        <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 flex flex-col items-center justify-center text-[9px] text-white p-1 transition-opacity rounded">
                          <span className="truncate w-full text-center">{item.name.substring(0, 8)}...</span>
                          <span className={scoreClass}>{item.aiScore}%</span>
                        </div>
                      </div>
                    );
                  })
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
                      キュー ({queue.length}件)
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2 flex-1 content-start overflow-y-auto max-h-52 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
                    {queue.map((item) => (
                      <div
                        key={item.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedQueueId(selectedQueueId === item.id ? null : item.id);
                          setSelectedHistoryId(null); // Clear history selection
                        }}
                        className={`queue-item relative group cursor-pointer ${item.status === "processing" ? "active" :
                          item.status === "ai" ? "result-ai" :
                            item.status === "unknown" ? "result-unknown" :
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

            {/* URL Input - 将来のBot連携用に非表示 */}
            {false && (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !isLoadingUrl && handleUrlSubmit()}
                  placeholder="画像URLを貼り付け（Twitter/Pixiv等）"
                  className="flex-1 px-3 py-2 rounded-lg bg-card-bg border border-border text-text-primary placeholder-muted text-sm focus:outline-none focus:border-accent"
                  disabled={isLoadingUrl}
                />
                <button
                  onClick={handleUrlSubmit}
                  disabled={!urlInput.trim() || isLoadingUrl || backendStatus === "offline"}
                  className="px-4 py-2 rounded-lg bg-accent/20 border border-accent text-accent font-medium text-sm hover:bg-accent/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoadingUrl ? "読込中..." : "URL解析"}
                </button>
              </div>
            )}

            {/* Buttons */}
            <div className="flex gap-3">
              <button
                onClick={startBatchScan}
                disabled={!canExecute}
                className="flex-grow primary-button flex items-center justify-center gap-2"
              >
                <Search className="w-4 h-4" />
                <span>スキャン開始</span>
                <span className="font-normal">
                  - 残り{(authUser?.isAdmin || rateLimitRemaining === -1) ? "∞" : (rateLimitRemaining ?? "--")}/{(authUser?.isAdmin || rateLimitRemaining === -1) ? "∞" : (authUser?.isVip ? "240" : "24")}枚
                </span>
                {!authUser?.isAdmin && rateLimitRemaining !== -1 && (
                  <span className="text-xs opacity-70 font-normal">
                    (午前0時にリセット)
                  </span>
                )}
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
          <p><a href="/disclaimer" className="hover:underline">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="hover:underline">お問い合わせ</a></p>
        </div>
      </footer>

      {/* VIP Modal */}
      <VipModal
        isOpen={isVipModalOpen}
        onClose={() => setIsVipModalOpen(false)}
        authUser={authUser}
        feature="checker"
      />
    </div>
  );
}
