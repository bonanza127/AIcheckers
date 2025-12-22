"use client";

import { useState } from "react";
import { X, ExternalLink, RefreshCw, Check, AlertCircle } from "lucide-react";

type VipModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://api.aicheckers.net";

export default function VipModal({ isOpen, onClose }: VipModalProps) {
  const [pixivId, setPixivId] = useState("");
  const [verifyStatus, setVerifyStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [verifyMessage, setVerifyMessage] = useState("");

  // 直接支援用
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // ログイン用
  const [loginId, setLoginId] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  if (!isOpen) return null;

  const handleFanboxClick = () => {
    window.open("https://www.fanbox.cc/@aicheckers", "_blank");
  };

  const handleVerify = async () => {
    if (!pixivId.trim()) {
      setVerifyStatus("error");
      setVerifyMessage("pixiv IDを入力してください");
      return;
    }

    setVerifyStatus("loading");
    setVerifyMessage("");

    try {
      const response = await fetch(`${API_BASE}/verify-fanbox`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixiv_id: pixivId.trim() }),
      });

      const data = await response.json();

      if (data.status === "success" || data.status === "already_vip") {
        setVerifyStatus("success");
        setVerifyMessage(data.message);
        localStorage.setItem("vip_pixiv_id", pixivId.trim());
      } else {
        setVerifyStatus("error");
        setVerifyMessage(data.message || "確認できませんでした");
      }
    } catch {
      setVerifyStatus("error");
      setVerifyMessage("通信エラーが発生しました");
    }
  };

  const handleDirectRegister = (e: React.FormEvent) => {
    e.preventDefault();
    alert("登録機能は準備中です");
  };

  const handleOAuthRegister = (provider: string) => {
    alert(`${provider}での登録機能は準備中です`);
  };

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    alert("ログイン機能は準備中です");
  };

  const handleOAuthLogin = (provider: string) => {
    alert(`${provider}でのログイン機能は準備中です`);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-card-bg border border-gray-700 rounded-2xl shadow-2xl max-w-3xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-muted hover:text-white transition-colors z-10"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="p-6 pb-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold bg-gradient-to-r from-amber-400 to-yellow-300 bg-clip-text text-transparent">
              VIP登録
            </span>
            <span className="text-muted">-</span>
            <span className="text-muted text-sm">プレミアム機能をアンロック</span>
          </div>
        </div>

        {/* メインコンテンツ */}
        <div className="p-6 space-y-6">

          {/* VIP特典 */}
          <div className="bg-gradient-to-br from-amber-500/10 to-yellow-500/10 rounded-lg p-4 border border-amber-500/30">
            <p className="text-sm font-bold text-amber-400 mb-3">VIP特典</p>
            <div className="flex flex-wrap gap-6">
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 text-success flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <div>
                  <span className="font-medium text-sm">スキャン回数アップ</span>
                  <div className="text-muted text-xs">
                    <span className="line-through">24枚/日</span>
                    <span className="text-amber-400 font-bold ml-1">→ 500枚/日</span>
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 text-success flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <div>
                  <span className="font-medium text-sm">最新モデル</span>
                  <div className="text-muted text-xs">ベータ版先行利用</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 text-success flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <div>
                  <span className="font-medium text-sm">広告非表示</span>
                  <div className="text-muted text-xs">快適な利用体験</div>
                </div>
              </div>
              <div className="flex items-baseline gap-1 ml-auto">
                <span className="text-2xl font-extrabold bg-gradient-to-r from-amber-400 to-yellow-300 bg-clip-text text-transparent">
                  ¥300
                </span>
                <span className="text-muted text-sm">/ 月額</span>
              </div>
            </div>
          </div>

          {/* 登録方法: 左右2カラム */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">

            {/* 左: FANBOX連携 */}
            <div className="space-y-3 p-4 bg-deep-bg rounded-lg border border-gray-700 flex flex-col">
              <h3 className="font-bold text-sm flex items-center gap-2 text-amber-400">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
                </svg>
                pixiv FANBOXで支援
              </h3>
              {/* Step 1 */}
              <div className="flex items-center gap-2 text-sm text-muted">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-bold">1</span>
                <span>FANBOXで支援プランに加入</span>
              </div>
              <button
                onClick={handleFanboxClick}
                className="w-full py-2.5 rounded-lg bg-[#F5A623] hover:bg-[#E09620] transition-colors flex items-center justify-center gap-2"
              >
                <span className="font-bold text-black text-sm">FANBOXページを開く</span>
                <ExternalLink className="w-4 h-4 text-black" />
              </button>

              {/* Step 2 */}
              <div className="flex items-center gap-2 text-sm text-muted">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-bold">2</span>
                <span>pixiv IDを入力してVIPステータスを取得</span>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={pixivId}
                  onChange={(e) => setPixivId(e.target.value)}
                  placeholder="pixiv ID"
                  className="w-1/2 px-3 py-2 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors text-sm"
                />
                <button
                  onClick={handleVerify}
                  disabled={verifyStatus === "loading"}
                  className="px-4 py-2 rounded-lg font-bold bg-amber-500 hover:bg-amber-400 text-black transition-all text-sm flex items-center gap-2 disabled:opacity-50"
                >
                  {verifyStatus === "loading" ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <RefreshCw className="w-4 h-4" />
                  )}
                  同期
                </button>
              </div>
              <p className="text-xs text-muted pl-7 mt-auto">
                ※ IDはpixiv.net/users/<span className="text-amber-400">数字</span>の数字部分
              </p>

              {verifyStatus === "success" && (
                <div className="flex items-center gap-2 text-success text-sm bg-success/10 p-2.5 rounded-lg">
                  <Check className="w-4 h-4" />
                  {verifyMessage}
                </div>
              )}
              {verifyStatus === "error" && (
                <div className="flex items-center gap-2 text-danger text-sm bg-danger/10 p-2.5 rounded-lg">
                  <AlertCircle className="w-4 h-4" />
                  {verifyMessage}
                </div>
              )}
            </div>

            {/* 右: 直接支援 */}
            <div className="space-y-3 p-4 bg-deep-bg rounded-lg border border-gray-700">
              <h3 className="font-bold text-sm text-amber-400">直接支援して登録</h3>
              <form onSubmit={handleDirectRegister} className="space-y-2">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="メールアドレス"
                  className="w-full px-3 py-2 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors text-sm"
                />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="パスワード"
                  className="w-full px-3 py-2 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors text-sm"
                />
                <button
                  type="submit"
                  className="w-full py-2.5 rounded-lg font-bold bg-gradient-to-r from-amber-500 to-yellow-500 text-black hover:from-amber-400 hover:to-yellow-400 transition-all text-sm"
                >
                  登録して支払いへ
                </button>
              </form>

              {/* OAuth連携ボタン */}
              <div className="flex gap-2">
                <button
                  onClick={() => handleOAuthRegister("Google")}
                  className="flex-1 py-2 rounded-lg bg-white hover:bg-gray-100 transition-colors flex items-center justify-center gap-1.5"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  <span className="text-xs font-bold text-gray-700">Google</span>
                </button>
                <button
                  onClick={() => handleOAuthRegister("Twitter")}
                  className="flex-1 py-2 rounded-lg bg-black hover:bg-gray-900 transition-colors flex items-center justify-center gap-1.5 border border-gray-600"
                >
                  <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                  </svg>
                  <span className="text-xs font-bold text-white">X</span>
                </button>
                <button
                  onClick={() => handleOAuthRegister("PayPal")}
                  className="flex-1 py-2 rounded-lg bg-[#0070BA] hover:bg-[#005C99] transition-colors flex items-center justify-center gap-1.5"
                >
                  <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7.076 21.337H2.47a.641.641 0 0 1-.633-.74L4.944.901C5.026.382 5.474 0 5.998 0h7.46c2.57 0 4.578.543 5.69 1.81 1.01 1.15 1.304 2.42 1.012 4.287-.023.143-.047.288-.077.437-.983 5.05-4.349 6.797-8.647 6.797h-2.19c-.524 0-.968.382-1.05.9l-1.12 7.106zm14.146-14.42a3.35 3.35 0 0 0-.607-.541c1.27 4.93-1.066 7.627-6.14 7.627H12.28a1.284 1.284 0 0 0-1.268 1.086l-1.02 6.477-.293 1.86a.642.642 0 0 0 .634.74h3.698c.456 0 .846-.334.918-.786l.038-.194.726-4.614.047-.254a.924.924 0 0 1 .913-.786h.58c3.74 0 6.67-1.52 7.525-5.92.357-1.84.172-3.372-.774-4.452a3.75 3.75 0 0 0-.937-.723z"/>
                  </svg>
                  <span className="text-xs font-bold text-white">PayPal</span>
                </button>
              </div>
            </div>
          </div>

          {/* 既存VIPログイン */}
          <div className="border-t border-gray-700 pt-4">
            <h3 className="text-sm font-bold mb-3 flex items-center gap-2 text-muted">
              <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
              </svg>
              既にVIP会員の方
            </h3>
            <form onSubmit={handleLogin} className="space-y-3">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={loginId}
                  onChange={(e) => setLoginId(e.target.value)}
                  placeholder="メールアドレス or pixiv ID"
                  className="flex-1 px-3 py-2 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors text-sm"
                />
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="パスワード"
                  className="flex-1 px-3 py-2 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors text-sm"
                />
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg font-bold border border-amber-500 text-amber-400 hover:bg-amber-500/10 transition-all text-sm"
                >
                  ログイン
                </button>
              </div>

              {/* OAuthログインボタン */}
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => handleOAuthLogin("Google")}
                  className="flex-1 py-1.5 rounded bg-gray-800 hover:bg-gray-700 transition-colors flex items-center justify-center gap-1 border border-gray-600"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  <span className="text-xs text-gray-400">Google</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleOAuthLogin("Twitter")}
                  className="flex-1 py-1.5 rounded bg-gray-800 hover:bg-gray-700 transition-colors flex items-center justify-center gap-1 border border-gray-600"
                >
                  <svg className="w-3.5 h-3.5 text-white" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                  </svg>
                  <span className="text-xs text-gray-400">X</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleOAuthLogin("pixiv")}
                  className="flex-1 py-1.5 rounded bg-gray-800 hover:bg-gray-700 transition-colors flex items-center justify-center gap-1 border border-gray-600"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="#0096FA">
                    <path d="M4.935 0A4.924 4.924 0 0 0 0 4.935v14.13A4.924 4.924 0 0 0 4.935 24h14.13A4.924 4.924 0 0 0 24 19.065V4.935A4.924 4.924 0 0 0 19.065 0zm7.81 4.547c2.181 0 4.058.676 5.399 1.847a6.118 6.118 0 0 1 2.116 4.66c.005 1.854-.88 3.476-2.257 4.563-1.375 1.092-3.225 1.697-5.258 1.697-2.314 0-4.46-.842-4.46-.842v2.718c.397.116 1.048.365.635.365H5.456c-.41 0-.064-.249.384-.365V7.682c0-.235-.173-.413-.173-.413h3.544s.173.178.173.413v.563s2.001-.698 3.361-.698zm.062 1.349c-1.479-.004-2.722.494-3.606 1.425-.886.935-1.327 2.155-1.327 3.556 0 1.397.441 2.518 1.327 3.416.882.899 2.127 1.37 3.606 1.365 1.483-.004 2.724-.466 3.606-1.365.886-.898 1.327-2.019 1.327-3.416 0-1.401-.441-2.621-1.327-3.556-.882-.931-2.123-1.429-3.606-1.425z"/>
                  </svg>
                  <span className="text-xs text-gray-400">pixiv</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
