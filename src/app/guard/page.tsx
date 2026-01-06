"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { flushSync } from "react-dom";
import { Upload, Trash2, Cpu, Shield, History, Plus, Download, ChevronLeft, ChevronRight } from "lucide-react";
import VipModal from "@/components/VipModal";
import HamburgerMenu from "@/components/HamburgerMenu";

// API URL: 本番環境では api.aicheckers.net を使用
const getApiUrl = () => {
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    if (hostname === "aicheckers.net" || hostname === "www.aicheckers.net" || hostname.endsWith(".vercel.app")) {
      return "https://api.aicheckers.net";
    }
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
};

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
};

type HistoryItem = {
  id: string;
  name: string;
  preview: string; // 元画像のプレビュー
  protectedImage: string; // 保護済み画像（Base64 PNG）
  ssim: number;
  processingTime: number;
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
  const [currentProtectedImage, setCurrentProtectedImage] = useState<string | null>(null);
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [modalImageIndex, setModalImageIndex] = useState<0 | 1>(0); // 0: Original, 1: Protected
  // Guard モードでは DetectionResult は使用しない（保護専用）
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [isScanning, setIsScanning] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [strength, setStrength] = useState(0.6);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  // ガードモードではヒートマップ不使用
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [selectedModel, setSelectedModel] = useState<"anixplore" | "legekka" | "dinov3">("dinov3");
  // Guard モードでは URL 入力は使用しない
  const urlInput = "";
  const isLoadingUrl = false;
  const [rateLimitRemaining, setRateLimitRemaining] = useState<number | null>(null);
  const [guardProgress, setGuardProgress] = useState({ current: 0, total: 0 });
  const [timeUntilReset, setTimeUntilReset] = useState("--:--:--");
  const [isVipModalOpen, setIsVipModalOpen] = useState(false);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true); // 認証状態確認中

  const fileInputRef = useRef<HTMLInputElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

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
        message: `🔑 Developer access granted.Welcome, ${name} !`
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
        fetch(`${apiUrl} /auth/me`, {
          headers: { Authorization: `Bearer ${savedToken} ` }
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

      if (token && name && email) {
        setAuthUser({ name, email, token, isVip });
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
        fetch(`${apiUrl} /auth/me`, {
          headers: { Authorization: `Bearer ${savedToken} ` }
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

  // バックエンドの接続状態を監視
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const apiUrl = getApiUrl();
        const response = await fetch(`${apiUrl}/health`, {
          method: "GET",
          signal: AbortSignal.timeout(3000)
        });
        if (response.ok) {
          const data = await response.json();
          // Moonlight (status: "healthy") or "online"
          setBackendOnline(data.status === "healthy" || data.status === "online");
          // Guard用残りトークン数を取得 (なければChecker用)
          if (data.rate_limit?.guard_remaining !== undefined) {
            setRateLimitRemaining(data.rate_limit.guard_remaining);
          } else if (data.rate_limit?.remaining !== undefined) {
            setRateLimitRemaining(data.rate_limit.remaining);
          }
        } else {
          setBackendOnline(false);
        }
      } catch {
        setBackendOnline(false);
      }
    };

    // 初回チェック
    checkBackendHealth();

    // 10秒ごとにチェック
    const interval = setInterval(checkBackendHealth, 10000);
    return () => clearInterval(interval);
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
      const reader = new FileReader();
      reader.onload = (e) => {
        const newItem: QueueItem = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}-${file.name}`,
          name: file.name,
          preview: e.target?.result as string,
          status: "wait"
        };
        setQueue(prev => [...prev, newItem]);
      };
      reader.readAsDataURL(file);
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

  // Guard モードでは URL 解析機能は使用しない（ダミー関数）
  const handleUrlSubmit = useCallback(async () => { }, []);

  const processFile = async (file: File, queueItemId: string, displayIndex: number, total: number) => {
    const fileStartTime = Date.now();

    setQueue(prev => prev.map(item =>
      item.id === queueItemId ? { ...item, status: "processing" as const } : item
    ));

    const reader = new FileReader();
    const imageUrl = await new Promise<string>((resolve) => {
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.readAsDataURL(file);
    });

    // 処理開始時にphaseをscanningに設定
    setPhase("scanning");
    setCurrentImage(imageUrl);
    setCurrentFileName(file.name);

    // ファイルサイズを取得
    const fileSizeKB = (file.size / 1024).toFixed(1);

    addLog(`防壁構築開始: [${displayIndex}/${total}] ${file.name}`, "heading");
    addLog(`画像データ読み込み完了 (${fileSizeKB}KB)`, "process");

    // API呼び出し（SSEストリーミング /guard-stream エンドポイント）
    const formData = new FormData();
    formData.append("file", file);
    formData.append("strength", strength.toString());

    let protectedImage: string | undefined;
    let processingTime: number = 0;
    let ssim: number;
    let rateLimitError = false;

    // 進捗リセット
    setGuardProgress({ current: 0, total: 0 });

    try {
      const apiUrl = getApiUrl();
      const headers: HeadersInit = {};
      if (authUser?.token) {
        headers["Authorization"] = `Bearer ${authUser.token}`;
      }

      const response = await fetch(`${apiUrl}/guard-stream`, {
        method: "POST",
        body: formData,
        headers,
      });

      if (!response.ok) {
        if (response.status === 429) {
          const errorData = await response.json().catch(() => ({ detail: "レート制限に達しました" }));
          throw new Error(`RATE_LIMITED:${errorData.detail || "レート制限に達しました"}`);
        }
        if (response.status === 503) {
          throw new Error("MoonKnightモジュールが利用できません");
        }
        throw new Error(`API error: ${response.status}`);
      }

      // SSEストリームを読み取り
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("ストリームを開始できませんでした");
      }

      let buffer = "";
      let lastProgress = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSEメッセージは \n\n で区切られる
        const messages = buffer.split("\n\n");
        // 最後のメッセージは不完全な可能性があるのでバッファに戻す
        buffer = messages.pop() || "";

        for (const message of messages) {
          const lines = message.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const jsonStr = line.slice(6);
                const data = JSON.parse(jsonStr);

                if (data.type === "progress") {
                  // flushSyncで即座にDOMを更新（Reactのバッチングを回避）
                  flushSync(() => {
                    setGuardProgress({ current: data.current, total: data.total });
                  });
                  // 進捗ログ（10ステップごと）
                  if (data.current % 10 === 0 && data.current !== lastProgress) {
                    addLog(`> Semantic Attack: ${data.current}/${data.total} iterations...`, "detail");
                    lastProgress = data.current;
                  }
                } else if (data.type === "complete") {
                  protectedImage = data.protected_image;
                  processingTime = data.processing_time;
                  ssim = data.ssim;

                  setGuardProgress({ current: data.iterations, total: data.iterations });
                  addLog(`> 品質検証: SSIM = ${ssim.toFixed(4)} (${ssim >= 0.95 ? "良好" : "許容範囲"})`, "process");
                  addLog("> MoonKnight保護完了", "process");
                  addLog("> 防壁構築完了 - 画像は保護されました", "result");
                } else if (data.type === "error") {
                  throw new Error(data.message);
                }
              } catch {
                // JSON パースエラー - ログ出力してスキップ
                console.error("SSE parse error for line:", line.substring(0, 100));
              }
            }
          }
        }
      }

      // ストリーム終了後、残りのバッファを処理
      if (buffer.trim()) {
        const lines = buffer.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === "complete") {
                protectedImage = data.protected_image;
                processingTime = data.processing_time;
                ssim = data.ssim;
                setGuardProgress({ current: data.iterations, total: data.iterations });
                addLog(`> 品質検証: SSIM = ${ssim.toFixed(4)} (${ssim >= 0.95 ? "良好" : "許容範囲"})`, "process");
                addLog("> MoonKnight保護完了", "process");
                addLog("> 防壁構築完了 - 画像は保護されました", "result");
              } else if (data.type === "error") {
                throw new Error(data.message);
              }
            } catch {
              // 無視
            }
          }
        }
      }

      if (!protectedImage) {
        throw new Error("保護画像の生成に失敗しました");
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      if (errorMessage.startsWith("RATE_LIMITED:")) {
        const detail = errorMessage.replace("RATE_LIMITED:", "");
        addLog(`ERROR: ${detail}`, "error");
        rateLimitError = true;
      } else {
        addLog(`ERROR: ${errorMessage}`, "error");
      }
      processingTime = (Date.now() - fileStartTime) / 1000;
      ssim = 0;
    }

    // キューのステータス更新
    const queueStatus = rateLimitError ? "unknown" as const : protectedImage ? "human" as const : "unknown" as const;
    setQueue(prev => prev.map(item =>
      item.id === queueItemId ? { ...item, status: queueStatus } : item
    ));

    // 履歴に追加（保護済み画像を含む）
    if (protectedImage) {
      setHistory(prev => [{
        id: `history-${Date.now()}`,
        name: file.name,
        preview: imageUrl,
        protectedImage: protectedImage,
        ssim,
        processingTime,
        timestamp: new Date()
      }, ...prev].slice(0, 100));
    }

    // Remove from queue after complete
    setQueue(prev => prev.filter(item => item.id !== queueItemId));

    setPhase("complete");
    // 結果をステートにセット
    if (protectedImage) {
      setCurrentProtectedImage(protectedImage);
    }
    const logMessage = rateLimitError
      ? "エラー: レート制限に達しました"
      : protectedImage
        ? `防壁構築完了: ${processingTime.toFixed(2)}秒`
        : "エラー: 防壁構築に失敗しました";
    addLog(logMessage, rateLimitError || !protectedImage ? "error" : "result");

    return { rateLimitError };
  };

  const startBatchScan = async () => {
    const files = (window as unknown as { _pendingFiles?: File[] })._pendingFiles || [];
    const queueSnapshot = [...queue]; // Take snapshot of queue IDs
    if (isScanning || files.length === 0) return;

    setIsScanning(true);
    setStartTime(Date.now());
    setElapsedTime(0);
    setBatchProgress({ current: 0, total: files.length });
    addLog("--- BATCH SCAN INITIATED (一括解析開始) ---", "heading");

    let completedCount = 0;
    for (let i = 0; i < files.length; i++) {
      setBatchProgress({ current: i + 1, total: files.length });
      const result = await processFile(files[i], queueSnapshot[i]?.id || "", i + 1, files.length);

      // レート制限に達したらバッチを中断
      if (result?.rateLimitError) {
        const remaining = files.length - i - 1;
        addLog(`--- BATCH SCAN ABORTED (レート制限により中断) ---`, "error");
        addLog(`STATUS: ${remaining}個のアーティファクトが未処理です。しばらく待ってから再試行してください。`, "error");
        break;
      }
      completedCount++;
    }

    setIsScanning(false);
    setStartTime(null);

    if (completedCount === files.length) {
      addLog("--- BATCH SCAN COMPLETE (一括解析完了) ---", "heading");
      addLog(`STATUS: 全ての${files.length}個のアーティファクトの処理が完了しました。`, "process");
    }

    // Clear pending files
    (window as unknown as { _pendingFiles: File[] })._pendingFiles = [];
  };

  const resetUI = () => {
    if (isScanning) return;

    setQueue([]);
    setCurrentImage(null);
    setCurrentFileName("");
    setPhase("idle");
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

  // X（Twitter）に結果を共有（Guard モード用）
  const shareToX = () => {
    if (phase !== "complete" || history.length === 0) return;

    const text = `【AI学習防止ガード】
🛡️ PROTECTED
MoonKnight V3 (旧FastProtect) で画像を保護しました

#AIイラストガード #aicheckers`;

    const shareUrl = "https://aicheckers.net/guard";
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareUrl)}`;

    window.open(twitterUrl, "_blank", "width=550,height=420");
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
    // スキャン中は常にPROCESSING...を表示
    if (phase === "scanning") {
      return { text: "構築中...", className: "verdict-loading" };
    }
    // スキャン完了時のみ結果を表示
    if (phase === "complete") {
      // ガードページでは常に「完了」表示（紫色）
      return { text: "PROTECTED", className: "verdict-protected" };
    }
    return { text: "待機中", className: "verdict-pending" };
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
    setCurrentImage(item.preview);
    setCurrentProtectedImage(item.protectedImage);
    setCurrentFileName(item.name);
    setPhase("complete");
  };

  // 個別ダウンロード
  const downloadSingleImage = (item: HistoryItem) => {
    if (!item.protectedImage) return;

    const link = document.createElement("a");
    link.href = item.protectedImage.startsWith("data:") ? item.protectedImage : `data:image/png;base64,${item.protectedImage}`;
    // ファイル名から拡張子を除去して _protected.png を付与
    const baseName = item.name.replace(/\.[^/.]+$/, "");
    link.download = `${baseName}_protected.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addLog(`ダウンロード: ${link.download}`, "info");
  };

  const handleDownloadCurrent = () => {
    if (!currentProtectedImage || !currentFileName) return;
    const link = document.createElement("a");
    link.href = currentProtectedImage.startsWith("data:") ? currentProtectedImage : `data:image/png;base64,${currentProtectedImage}`;
    const baseName = currentFileName.replace(/\.[^/.]+$/, "");
    link.download = `${baseName}_protected.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addLog(`ダウンロード: ${link.download}`, "info");
  };

  const openImageModal = (index: 0 | 1) => {
    if (!currentImage) return;
    setModalImageIndex(index);
    setIsImageModalOpen(true);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isImageModalOpen) return;
      if (e.key === "ArrowLeft") setModalImageIndex(0);
      if (e.key === "ArrowRight" && currentProtectedImage) setModalImageIndex(1);
      if (e.key === "Escape") setIsImageModalOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isImageModalOpen, currentProtectedImage]);

  // まとめてダウンロード（ZIP形式）
  const downloadAllImages = async () => {
    const protectedItems = history.filter(item => item.protectedImage);
    if (protectedItems.length === 0) {
      addLog("ダウンロードできる画像がありません", "error");
      return;
    }

    addLog(`一括ダウンロード開始: ${protectedItems.length}件`, "process");

    // JSZipを動的にインポート（CDNから）
    try {
      const JSZip = (await import("jszip")).default;
      const zip = new JSZip();

      protectedItems.forEach((item, index) => {
        const baseName = item.name.replace(/\.[^/.]+$/, "");
        const fileName = `${baseName}_protected.png`;
        // Base64をバイナリに変換
        const binaryString = atob(item.protectedImage);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        zip.file(fileName, bytes);
      });

      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `protected_images_${new Date().toISOString().slice(0, 10)}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      addLog(`一括ダウンロード完了: ${protectedItems.length}件をZIPで保存`, "result");
    } catch {
      // JSZipがない場合は個別にダウンロード
      addLog("ZIP作成に失敗。個別ダウンロードを実行中...", "info");
      protectedItems.forEach((item, index) => {
        setTimeout(() => downloadSingleImage(item), index * 500);
      });
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="site-header sticky top-0 z-40 p-4">
        <div className="container mx-auto flex justify-between items-center">
          {/* 左: ロゴ + メニュー */}
          <div className="flex items-center gap-0">
            <HamburgerMenu variant="guard" />
            <img src="/logo-transparent.png" alt="AI Checkers" className="w-14 h-14" />
            <h2 className="text-2xl font-bold tracking-tight">
              AIイラストガード
              <span className="text-sm font-light text-muted">　//　</span>
              <a
                href="/guard/how-it-works"
                className="text-sm font-medium text-muted hover:text-accent hover:bg-accent/5 px-2 py-1 rounded transition-all border border-transparent hover:border-accent/30"
              >
                How it works?
              </a>
            </h2>
          </div>

          {/* 右: ステータス + VIP */}
          <div className="flex items-center gap-4 text-xs">
            {/* Server Status */}
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline === null ? "bg-gray-500 animate-pulse" : backendOnline ? "bg-success" : "bg-danger"}`} />
              <span className="text-muted">Server Status:</span>
              <span className={backendOnline ? "text-success" : backendOnline === false ? "text-danger" : "text-gray-500"}>
                {backendOnline === null ? "..." : backendOnline ? "Online" : "Offline"}
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
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold mb-3 tracking-tight whitespace-nowrap">AIイラストガードがあなたの作品を無断学習から守ります</h1>
          <p className="text-muted text-lg">
            人間の目には見えないノイズを混ぜることにより、作風の模倣を防ぐAIポイズニングの最新版。<br />
            複数の技法をかけ合わせた<span className="text-accent font-bold">MoonKnight V3</span>により、生成AIの学習を顕著に妨害します。
          </p>
          <p className="text-xs text-muted mt-2">
            ※ 作品の質を下げないよう、防壁は最小限のノイズで構成されています
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
                    <div className="flex gap-4 w-full">
                      {/* Left: Original */}
                      <div className="flex-1 min-w-0 flex flex-col items-center">
                        <div
                          className={`relative w-full h-72 flex items-center justify-center bg-black/20 rounded-lg border border-white/5 overflow-hidden cursor-zoom-in group ${phase === "scanning" ? "scanning" : ""}`}
                          onClick={() => openImageModal(0)}
                        >
                          <img
                            src={previewImage}
                            alt="Original"
                            className="max-w-full max-h-full object-contain"
                          />
                          <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 text-xs text-white rounded">Before</div>
                        </div>
                      </div>

                      {/* Right: Protected */}
                      <div className="flex-1 min-w-0 flex flex-col items-center">
                        {currentProtectedImage ? (
                          <div
                            className="relative w-full h-72 flex items-center justify-center bg-black/20 rounded-lg border border-accent/30 overflow-hidden cursor-zoom-in group"
                            onClick={() => openImageModal(1)}
                          >
                            <div className="absolute inset-0 bg-accent/5 pointer-events-none"></div>
                            <img
                              src={currentProtectedImage.startsWith("data:") ? currentProtectedImage : `data:image/png;base64,${currentProtectedImage}`}
                              alt="Protected"
                              className="max-w-full max-h-full object-contain"
                            />
                            <div className="absolute top-2 left-2 px-2 py-1 bg-accent text-xs text-white font-bold rounded shadow-lg">After</div>

                            {/* Download Button */}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDownloadCurrent();
                              }}
                              className="absolute top-2 right-2 flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold bg-white text-accent hover:bg-gray-100 rounded-md shadow-xl transition-all opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0"
                              title="画像をダウンロード"
                            >
                              <Download className="w-3.5 h-3.5" />
                              Save
                            </button>
                          </div>
                        ) : (
                          <div className="w-full h-72 flex flex-col items-center justify-center bg-black/20 rounded-lg border border-dashed border-gray-700">
                            {phase === "scanning" ? (
                              <div className="flex flex-col items-center animate-pulse">
                                <Shield className="w-8 h-8 text-accent mb-2 opacity-50" />
                                <span className="text-xs text-accent">Protecting...</span>
                              </div>
                            ) : (
                              <span className="text-sm text-dim">Pending...</span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="scan-placeholder w-full h-72 flex flex-col items-center justify-center">
                      <Cpu className="w-12 h-12 text-dim mb-2" />
                      <p className="text-muted font-light">SYSTEM READY FOR INFERENCE</p>
                    </div>
                  )}
                  {previewFileName && (
                    <div className="flex items-center justify-center gap-2 mt-3">
                      <p className="text-sm text-muted truncate font-mono">{previewFileName}</p>
                      {phase === "complete" && <span className="text-xs text-accent font-semibold">[PROTECTED]</span>}
                    </div>
                  )}
                </div>

                {/* Console Log */}
                <div className="w-full md:w-1/2">
                  <div
                    ref={logContainerRef}
                    className="console-log h-72 overflow-y-auto"
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
                disabled={phase !== "complete" || history.length === 0}
                className="absolute top-3 right-3 p-1.5 text-muted hover:text-white hover:drop-shadow-[0_0_6px_rgba(255,255,255,0.5)] disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                title="Xで結果を共有"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </button>
              <h3 className="text-xl font-bold border-b-2 border-accent pb-2 mb-4 tracking-wide">
                防壁構築
              </h3>

              {/* Row 1: Batch Status + Model + Logic + Processing Time */}
              <div className="flex flex-wrap justify-between items-center mb-4 text-sm text-muted gap-2">
                <span>BATCH STATUS: {batchProgress.current || "-"} / {batchProgress.total || "-"}</span>
                <span>使用モデル: <span className="text-accent font-bold">MoonKnight V3</span></span>
                <span>ロジック: DWT + 知覚マスキング</span>
                <span>PROCESSING TIME: <span className="font-bold">{elapsedTime.toFixed(2)}s</span></span>
              </div>

              {/* Progress Bar - MoonKnightと同期 */}
              <div className="mb-6">
                <div className="flex justify-between text-base mb-1">
                  <span className="font-semibold uppercase text-accent">
                    {phase === "scanning" && guardProgress.total > 0
                      ? `Processing: ${guardProgress.current}/${guardProgress.total}`
                      : "Now loading"}
                  </span>
                  <span className="font-bold text-accent">
                    {phase === "complete"
                      ? "100"
                      : phase === "scanning" && guardProgress.total > 0
                        ? Math.round((guardProgress.current / guardProgress.total) * 100)
                        : 0}%
                  </span>
                </div>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill guard-progress"
                    style={{
                      width: `${phase === "complete"
                        ? 100
                        : phase === "scanning" && guardProgress.total > 0
                          ? (guardProgress.current / guardProgress.total) * 100
                          : 0}%`
                    }}
                  />
                </div>
              </div>

              {/* Status Result */}
              <div className="flex justify-between items-end border-t border-gray-700 pt-4">
                <span className="text-2xl font-medium text-muted uppercase">STATUS:</span>
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
                <div className="flex items-center gap-2">
                  {history.length > 0 && (
                    <button
                      onClick={downloadAllImages}
                      className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-accent hover:bg-accent/10 rounded transition-colors border border-accent/30 hover:border-accent"
                      title="すべてダウンロード (ZIP)"
                    >
                      <Download className="w-3 h-3" />
                      一括DL
                    </button>
                  )}
                  <span className="text-sm font-normal text-muted">({history.length}件)</span>
                </div>
              </h3>
              <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto p-1 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
                {history.length === 0 ? (
                  <p className="text-muted text-sm italic">保護履歴はありません。</p>
                ) : (
                  history.map((item) => (
                    <div
                      key={item.id}
                      className={`history-item relative group cursor-pointer border-accent shadow-[0_0_8px_rgba(139,92,246,0.4)] ${selectedHistoryId === item.id ? "ring-2 ring-accent ring-offset-2 ring-offset-card-bg" : ""}`}
                      title={`${item.name} - 保護完了`}
                    >
                      <img
                        src={item.preview}
                        alt={item.name}
                        onClick={() => handleHistoryClick(item)}
                        className="w-12 h-12 object-cover rounded cursor-pointer"
                      />
                      {/* 完了ラベル - 紫 */}
                      <div className="absolute -top-1 -right-1 px-1.5 py-0.5 text-[8px] font-bold rounded bg-gradient-to-r from-purple-500 to-violet-400 text-white shadow-md">
                        完
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
                  {/* Slider for strength, placed after the queue items */}
                  <div className="mt-4">
                    <label htmlFor="strength-slider" className="block text-sm font-medium text-muted mb-2">
                      保護強度: <span className="font-bold text-accent">{(strength * 100).toFixed(0)}%</span>
                    </label>
                    <input
                      id="strength-slider"
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.01"
                      value={strength}
                      onChange={(e) => setStrength(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-dim mt-1">
                      <span>Weak (0.1)</span>
                      <span className="text-accent font-bold">Standard (0.6)</span>
                      <span>Strong (1.0)</span>
                    </div>
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
                  onChange={() => { }}
                  onKeyDown={(e) => e.key === "Enter" && !isLoadingUrl && handleUrlSubmit()}
                  placeholder="画像URLを貼り付け（Twitter/Pixiv等）"
                  className="flex-1 px-3 py-2 rounded-lg bg-card-bg border border-border text-text-primary placeholder-muted text-sm focus:outline-none focus:border-accent"
                  disabled={isLoadingUrl}
                />
                <button
                  onClick={handleUrlSubmit}
                  disabled={!urlInput.trim() || isLoadingUrl || !backendOnline}
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
                <Shield className="w-4 h-4" />
                <span>防壁を構築</span>
                <span className="font-normal">
                  - 残り{(authUser?.isAdmin || rateLimitRemaining === -1) ? "∞" : (rateLimitRemaining ?? "--")}/{(authUser?.isAdmin || rateLimitRemaining === -1) ? "∞" : (authUser?.isVip ? "30" : "3")}枚
                </span>
                {!authUser?.isAdmin && rateLimitRemaining !== -1 && (
                  <span className="text-xs opacity-70 font-normal">
                    (8時間刻みで{authUser?.isVip ? "5" : "1"}枚回復)
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
          <p><a href="/guard/disclaimer" className="hover:text-foreground transition-colors">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="hover:underline">お問い合わせ</a></p>
        </div>
      </footer>

      {/* VIP Modal */}
      <VipModal
        isOpen={isVipModalOpen}
        onClose={() => setIsVipModalOpen(false)}
        authUser={authUser}
      />
      {/* Image Comparison Modal */}
      {isImageModalOpen && previewImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
          onClick={() => setIsImageModalOpen(false)}
        >
          <div className="flex items-center justify-center gap-4 max-w-[95vw] max-h-[95vh]" onClick={e => e.stopPropagation()}>
            {/* Left Arrow */}
            {currentProtectedImage && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setModalImageIndex(0);
                }}
                className={`p-3 rounded-full text-white transition-all backdrop-blur-sm ${modalImageIndex === 0 ? "bg-white/20 cursor-default opacity-50" : "bg-white/10 hover:bg-white/30"}`}
                aria-label="Previous image"
                disabled={modalImageIndex === 0}
              >
                <ChevronLeft className="w-8 h-8" />
              </button>
            )}

            {/* Center Content */}
            <div className="flex flex-col items-center gap-4">
              {/* Labels & Controls */}
              <div className="flex gap-4 bg-black/50 p-2 rounded-full backdrop-blur-md border border-white/10 z-10">
                <button
                  onClick={() => setModalImageIndex(0)}
                  className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${modalImageIndex === 0 ? "bg-white text-black shadow-lg" : "text-gray-400 hover:text-white"
                    }`}
                >
                  Before
                </button>
                <button
                  onClick={() => currentProtectedImage && setModalImageIndex(1)}
                  disabled={!currentProtectedImage}
                  className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${modalImageIndex === 1 ? "bg-accent text-white shadow-lg" : "text-gray-400 hover:text-accent disabled:opacity-30"
                    }`}
                >
                  After
                </button>
              </div>

              {/* Image Container */}
              <div className="relative max-h-full">
                <img
                  src={modalImageIndex === 0
                    ? previewImage
                    : (currentProtectedImage?.startsWith("data:") ? currentProtectedImage : `data:image/png;base64,${currentProtectedImage}`)
                  }
                  alt="Comparison View"
                  className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl"
                />

                {/* Close Button (Overlay on image) */}
                <button
                  onClick={() => setIsImageModalOpen(false)}
                  className="absolute top-4 right-4 p-2 bg-black/50 hover:bg-white/20 rounded-full text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Right Arrow */}
            {currentProtectedImage && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setModalImageIndex(1);
                }}
                className={`p-3 rounded-full text-white transition-all backdrop-blur-sm ${modalImageIndex === 1 ? "bg-white/20 cursor-default opacity-50" : "bg-white/10 hover:bg-white/30"}`}
                aria-label="Next image"
                disabled={modalImageIndex === 1}
              >
                <ChevronRight className="w-8 h-8" />
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
