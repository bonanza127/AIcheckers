"use client";

import { useState, useEffect } from "react";
import { X, CreditCard, UserPlus, LogIn, ArrowRight, CheckCircle, Loader2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://api.aicheckers.net";

type AuthUser = {
  name: string;
  email: string;
  token: string;
  isVip: boolean;
};

type VipModalProps = {
  isOpen: boolean;
  onClose: () => void;
  authUser?: AuthUser | null;
};

type ModalStep = "auth" | "payment" | "complete" | "status";

export default function VipModal({ isOpen, onClose, authUser }: VipModalProps) {
  const [step, setStep] = useState<ModalStep>("auth");

  // タブ切り替え（新規登録 / ログイン）
  const [authTab, setAuthTab] = useState<"register" | "login">("register");

  // 認証状態に応じてステップを設定
  useEffect(() => {
    if (authUser && isOpen) {
      setUserName(authUser.name);
      setUserEmail(authUser.email);
      if (authUser.isVip) {
        setStep("status"); // VIPならステータス表示
      } else {
        setStep("payment"); // 非VIPなら決済へ
      }
    } else if (isOpen && !authUser) {
      setStep("auth"); // 未ログインなら認証へ
    }
  }, [authUser, isOpen]);

  // 新規登録用
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");

  // ログイン用
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // ユーザー情報（ログイン後）
  const [userName, setUserName] = useState("");
  const [userEmail, setUserEmail] = useState("");

  // ローディング状態
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");

  if (!isOpen) return null;

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsProcessing(true);

    if (!email.trim()) {
      setError("メールアドレスを入力してください");
      setIsProcessing(false);
      return;
    }
    if (password.length < 8) {
      setError("パスワードは8文字以上で入力してください");
      setIsProcessing(false);
      return;
    }
    if (password !== passwordConfirm) {
      setError("パスワードが一致しません");
      setIsProcessing(false);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, name: email.split("@")[0] }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "登録に失敗しました");
      }

      // トークン保存
      localStorage.setItem("auth_token", data.token);
      setUserName(data.name);
      setUserEmail(data.email);
      setStep("payment");
    } catch (err) {
      setError(err instanceof Error ? err.message : "登録に失敗しました");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleOAuthRegister = (provider: string) => {
    // OAuthリダイレクト
    const endpoint = provider === "Google" ? "/auth/google" : "/auth/twitter";
    window.location.href = `${API_BASE}${endpoint}`;
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsProcessing(true);

    if (!loginEmail.trim()) {
      setError("メールアドレスを入力してください");
      setIsProcessing(false);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: loginEmail, password: loginPassword }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "ログインに失敗しました");
      }

      // トークン保存
      localStorage.setItem("auth_token", data.token);
      setUserName(data.name);
      setUserEmail(data.email);

      if (data.is_vip) {
        // VIPならページリロードしてステータス反映
        window.location.reload();
      } else {
        setStep("payment");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "ログインに失敗しました");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleOAuthLogin = (provider: string) => {
    // OAuthリダイレクト（登録と同じエンドポイント）
    const endpoint = provider === "Google" ? "/auth/google" : "/auth/twitter";
    window.location.href = `${API_BASE}${endpoint}`;
  };

  const handlePayment = async (method: "stripe" | "paypal" | "paypay") => {
    if (!userEmail) {
      setError("メールアドレスが設定されていません");
      return;
    }

    setIsProcessing(true);
    setError("");

    try {
      let endpoint = "/create-checkout-session";
      if (method === "paypal") {
        endpoint = "/create-paypal-payment";
      } else if (method === "paypay") {
        // PayPayは法人契約が必要なため、現在準備中
        setError("PayPay決済は現在準備中です");
        setIsProcessing(false);
        return;
      }

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: userEmail,
          payment_method: method,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "決済セッションの作成に失敗しました");
      }

      const data = await response.json();

      // 決済ページを新しいウィンドウで開く
      if (data.checkout_url) {
        window.open(data.checkout_url, "_blank", "width=500,height=700");
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : "エラーが発生しました");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClose = () => {
    // モーダルを閉じる時にリセット
    setStep("auth");
    setAuthTab("register");
    setEmail("");
    setPassword("");
    setPasswordConfirm("");
    setLoginEmail("");
    setLoginPassword("");
    setUserName("");
    setUserEmail("");
    setError("");
    setIsProcessing(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative bg-card-bg border border-gray-700 rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Close button */}
        <button
          onClick={handleClose}
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
                    <span className="text-amber-400 font-bold ml-1">→ 240枚/日</span>
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

          {/* STEP 1: 認証 + お支払い方法プレビュー */}
          {step === "auth" && (
            <>
              {/* エラー表示 */}
              {error && (
                <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              <div className="space-y-4 p-4 bg-deep-bg rounded-lg border border-gray-700">
                {/* タブ */}
                <div className="flex border-b border-gray-600">
                  <button
                    onClick={() => setAuthTab("register")}
                    className={`flex-1 py-2 text-sm font-bold flex items-center justify-center gap-1.5 border-b-2 transition-colors ${
                      authTab === "register"
                        ? "border-amber-400 text-amber-400"
                        : "border-transparent text-muted hover:text-white"
                    }`}
                  >
                    <UserPlus className="w-4 h-4" />
                    新規登録
                  </button>
                  <button
                    onClick={() => setAuthTab("login")}
                    className={`flex-1 py-2 text-sm font-bold flex items-center justify-center gap-1.5 border-b-2 transition-colors ${
                      authTab === "login"
                        ? "border-amber-400 text-amber-400"
                        : "border-transparent text-muted hover:text-white"
                    }`}
                  >
                    <LogIn className="w-4 h-4" />
                    ログイン
                  </button>
                </div>

                {/* 新規登録タブ */}
                {authTab === "register" && (
                  <div className="space-y-4">
                    {/* ステップインジケーター（認証ステップでは1がアクティブ） */}
                    <div className="flex items-center justify-center gap-2 text-xs pb-2">
                      <div className="flex items-center gap-1 text-amber-400">
                        <div className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold bg-amber-400 text-black">
                          1
                        </div>
                        <span>アカウント</span>
                      </div>
                      <ArrowRight className="w-3 h-3 text-gray-600" />
                      <div className="flex items-center gap-1 text-muted">
                        <div className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold bg-gray-700">
                          2
                        </div>
                        <span>お支払い</span>
                      </div>
                      <ArrowRight className="w-3 h-3 text-gray-600" />
                      <div className="flex items-center gap-1 text-muted">
                        <div className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold bg-gray-700">
                          3
                        </div>
                        <span>完了</span>
                      </div>
                    </div>

                    {/* OAuth登録ボタン */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleOAuthRegister("Google")}
                        className="flex-1 py-2.5 rounded-lg bg-white hover:bg-gray-200 active:bg-gray-300 transition-colors flex items-center justify-center gap-2 border-2 border-gray-300 hover:border-gray-400 active:scale-95"
                      >
                        <svg className="w-5 h-5" viewBox="0 0 24 24">
                          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                        </svg>
                        <span className="text-sm font-bold text-gray-700">Googleで登録</span>
                      </button>
                      <button
                        onClick={() => handleOAuthRegister("Twitter")}
                        className="flex-1 py-2.5 rounded-lg bg-black hover:bg-gray-900 active:bg-gray-800 transition-colors flex items-center justify-center gap-2 border-2 border-gray-600 hover:border-gray-500 active:scale-95"
                      >
                        <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                        </svg>
                        <span className="text-sm font-bold text-white">Xで登録</span>
                      </button>
                    </div>

                    {/* 区切り線 */}
                    <div className="flex items-center gap-2">
                      <div className="flex-1 border-t border-gray-600" />
                      <span className="text-xs text-muted">またはメールで登録</span>
                      <div className="flex-1 border-t border-gray-600" />
                    </div>

                    {/* メール登録フォーム */}
                    <form onSubmit={handleRegister} className="space-y-3">
                      <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="メールアドレス"
                        className="w-full px-4 py-2.5 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                      />
                      <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="パスワード（8文字以上）"
                        className="w-full px-4 py-2.5 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                      />
                      <input
                        type="password"
                        value={passwordConfirm}
                        onChange={(e) => setPasswordConfirm(e.target.value)}
                        placeholder="パスワード（確認）"
                        className="w-full px-4 py-2.5 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                      />
                      <button
                        type="submit"
                        className="w-full py-3 rounded-lg font-bold bg-gradient-to-r from-amber-500 to-yellow-500 text-black hover:from-amber-400 hover:to-yellow-400 transition-all flex items-center justify-center gap-2"
                      >
                        登録してお支払いへ
                        <ArrowRight className="w-4 h-4" />
                      </button>
                    </form>

                    {/* お支払い方法一覧（グレーアウト） */}
                    <div className="pt-4 border-t border-gray-700">
                      <h4 className="font-bold text-sm flex items-center gap-2 text-muted mb-1">
                        <CreditCard className="w-4 h-4" />
                        お支払い方法一覧
                      </h4>
                      <p className="text-xs text-muted mb-3">
                        外部サービスを使用しているため、サーバーに決済情報が残ることはありません
                      </p>
                      <div className="flex flex-col sm:flex-row gap-2 opacity-50">
                        <div className="flex-1 py-2 px-3 rounded-lg bg-gradient-to-b from-zinc-800 to-black border border-zinc-700 flex items-center justify-center gap-1.5 cursor-not-allowed">
                          <CreditCard className="w-3.5 h-3.5 text-gray-400" />
                          <span className="text-xs text-gray-400">Stripe(クレジットカード)</span>
                        </div>
                        <div className="flex-1 py-2 px-3 rounded-lg bg-[#003087]/50 border border-[#003087]/30 flex items-center justify-center cursor-not-allowed">
                          <span className="text-xs text-gray-400 font-bold">PayPal</span>
                        </div>
                        <div className="flex-1 py-2 px-3 rounded-lg bg-[#FF0033]/30 border border-[#FF0033]/30 flex items-center justify-center cursor-not-allowed">
                          <span className="text-xs text-gray-400 font-bold">PayPay</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* ログインタブ */}
                {authTab === "login" && (
                  <div className="space-y-4">
                    {/* OAuthログインボタン */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleOAuthLogin("Google")}
                        className="flex-1 py-2.5 rounded-lg bg-white hover:bg-gray-200 active:bg-gray-300 transition-colors flex items-center justify-center gap-2 border-2 border-gray-300 hover:border-gray-400 active:scale-95"
                      >
                        <svg className="w-5 h-5" viewBox="0 0 24 24">
                          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                        </svg>
                        <span className="text-sm font-bold text-gray-700">Googleでログイン</span>
                      </button>
                      <button
                        onClick={() => handleOAuthLogin("Twitter")}
                        className="flex-1 py-2.5 rounded-lg bg-black hover:bg-gray-900 active:bg-gray-800 transition-colors flex items-center justify-center gap-2 border-2 border-gray-600 hover:border-gray-500 active:scale-95"
                      >
                        <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                        </svg>
                        <span className="text-sm font-bold text-white">Xでログイン</span>
                      </button>
                    </div>

                    {/* 区切り線 */}
                    <div className="flex items-center gap-2">
                      <div className="flex-1 border-t border-gray-600" />
                      <span className="text-xs text-muted">またはメールでログイン</span>
                      <div className="flex-1 border-t border-gray-600" />
                    </div>

                    {/* メールログインフォーム */}
                    <form onSubmit={handleLogin} className="space-y-3">
                      <input
                        type="email"
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        placeholder="メールアドレス"
                        className="w-full px-4 py-2.5 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                      />
                      <input
                        type="password"
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        placeholder="パスワード"
                        className="w-full px-4 py-2.5 rounded-lg bg-card-bg border border-gray-600 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                      />
                      <button
                        type="submit"
                        className="w-full py-3 rounded-lg font-bold border border-amber-500 text-amber-400 hover:bg-amber-500/10 transition-all flex items-center justify-center gap-2"
                      >
                        ログイン
                        <ArrowRight className="w-4 h-4" />
                      </button>
                    </form>
                  </div>
                )}
              </div>
            </>
          )}

          {/* STEP 2: お支払い */}
          {step === "payment" && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-success" />
                <span className="text-muted">ようこそ、</span>
                <span className="text-amber-400 font-medium">{userName}</span>
                <span className="text-muted">さん</span>
              </div>

              <p className="text-sm text-muted">
                お支払い方法を選択してください。安全な決済サービスを利用しています。
              </p>

              {/* エラー表示 */}
              {error && (
                <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm p-3 rounded-lg">
                  {error}
                </div>
              )}

              {/* 決済方法ボタン */}
              <div className="grid grid-cols-1 gap-3">
                {/* クレジットカード（Stripe） */}
                <button
                  onClick={() => handlePayment("stripe")}
                  disabled={isProcessing}
                  className="group relative w-full py-4 px-5 rounded-xl bg-gradient-to-b from-zinc-800 via-zinc-900 to-black border border-zinc-700 hover:border-zinc-500 transition-all flex items-center gap-4 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="w-12 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-md flex items-center justify-center">
                    <CreditCard className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-bold text-white">クレジットカード</div>
                    <div className="text-xs text-muted">Visa / Mastercard / AMEX / JCB</div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-muted">Powered by</span>
                    <span className="text-sm font-bold text-[#6772E5]">stripe</span>
                  </div>
                </button>

                {/* PayPal */}
                <button
                  onClick={() => handlePayment("paypal")}
                  disabled={isProcessing}
                  className="group w-full py-4 px-5 rounded-xl bg-[#003087] hover:bg-[#001F5C] border border-[#003087] transition-all flex items-center gap-4 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="w-12 h-8 flex items-center justify-center">
                    <svg className="h-6" viewBox="0 0 100 26" fill="white">
                      <path d="M12.3 5.2h-6c-.4 0-.8.3-.9.7L3 20.6c-.1.3.2.6.5.6h2.9c.4 0 .8-.3.9-.7l.6-3.8c.1-.4.4-.7.9-.7h2c4.1 0 6.5-2 7.1-5.9.3-1.7 0-3-1-4-.9-.9-2.6-1.4-4.6-1.4zm.7 5.8c-.3 2.3-2 2.3-3.6 2.3h-.9l.7-4.2c0-.2.2-.4.4-.4h.4c1.1 0 2.1 0 2.7.6.3.4.4 1 .3 1.7z"/>
                      <path d="M35.2 10.9h-2.9c-.2 0-.4.2-.4.4l-.1.8-.2-.3c-.6-.9-2-1.2-3.4-1.2-3.2 0-5.9 2.4-6.4 5.8-.3 1.7.1 3.3 1.1 4.4.9 1 2.2 1.4 3.8 1.4 2.7 0 4.1-1.7 4.1-1.7l-.1.8c-.1.3.2.6.5.6h2.6c.4 0 .8-.3.9-.7l1.5-9.7c.1-.3-.2-.6-.5-.6zm-4.1 5.6c-.3 1.6-1.5 2.7-3.2 2.7-.8 0-1.5-.3-1.9-.7-.4-.5-.5-1.2-.4-2 .2-1.6 1.5-2.7 3.1-2.7.8 0 1.5.3 1.9.8.4.4.6 1.1.5 1.9z"/>
                      <path d="M55.6 10.9h-2.9c-.3 0-.5.1-.6.3l-3.6 5.3-1.5-5.1c-.1-.3-.4-.5-.8-.5h-2.8c-.4 0-.6.4-.5.7l2.9 8.4-2.7 3.8c-.3.4 0 .9.5.9h2.9c.3 0 .5-.1.6-.3l8.6-12.5c.2-.4-.1-.9-.6-.9z"/>
                    </svg>
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-bold text-white">PayPal</div>
                    <div className="text-xs text-blue-200">PayPalアカウントで簡単決済</div>
                  </div>
                </button>

                {/* PayPay */}
                <button
                  onClick={() => handlePayment("paypay")}
                  disabled={isProcessing}
                  className="group w-full py-4 px-5 rounded-xl bg-[#FF0033] hover:bg-[#E0002E] border border-[#FF0033] transition-all flex items-center gap-4 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="w-12 h-8 flex items-center justify-center">
                    <span className="text-white font-extrabold text-lg tracking-tight">Pay</span>
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-bold text-white">PayPay</div>
                    <div className="text-xs text-red-200">PayPayアプリで支払い</div>
                  </div>
                  <div className="bg-white/20 rounded px-2 py-0.5">
                    <span className="text-xs font-bold text-white">人気</span>
                  </div>
                </button>
              </div>

              {/* ローディング表示 */}
              {isProcessing && (
                <div className="flex items-center justify-center gap-2 text-amber-400">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span className="text-sm">決済ページを準備中...</span>
                </div>
              )}

              <p className="text-xs text-muted text-center">
                決済は安全なStripe / PayPalを通じて処理されます
              </p>
            </div>
          )}

          {/* STEP 3: 完了 */}
          {step === "complete" && (
            <div className="text-center space-y-4 py-4">
              <div className="w-16 h-16 bg-success/20 rounded-full flex items-center justify-center mx-auto">
                <CheckCircle className="w-10 h-10 text-success" />
              </div>
              <h3 className="text-xl font-bold text-white">VIP登録完了！</h3>
              <p className="text-muted">
                ご登録ありがとうございます。<br />
                VIP特典がすぐにご利用いただけます。
              </p>
              <button
                onClick={handleClose}
                className="px-6 py-2 rounded-lg font-bold bg-gradient-to-r from-amber-500 to-yellow-500 text-black hover:from-amber-400 hover:to-yellow-400 transition-all"
              >
                閉じる
              </button>
            </div>
          )}

          {/* VIPステータス表示 */}
          {step === "status" && authUser && (
            <div className="space-y-6">
              {/* ユーザー情報 */}
              <div className="bg-gradient-to-br from-amber-500/10 to-yellow-500/10 rounded-lg p-4 border border-amber-500/30">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-400 to-yellow-300 flex items-center justify-center text-black font-bold text-xl">
                    {authUser.name.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <p className="font-bold text-white">{authUser.name}</p>
                    <p className="text-sm text-muted">{authUser.email}</p>
                  </div>
                  <div className="ml-auto px-3 py-1 rounded-full bg-gradient-to-r from-amber-500 to-yellow-500 text-black text-xs font-bold">
                    VIP
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted">スキャン上限</p>
                    <p className="font-bold text-amber-400">240枚/日</p>
                  </div>
                  <div>
                    <p className="text-muted">回復レート</p>
                    <p className="font-bold text-amber-400">10枚/時間</p>
                  </div>
                </div>
              </div>

              {/* VIP特典一覧 */}
              <div className="space-y-2">
                <p className="text-sm font-bold text-muted">有効な特典</p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 rounded bg-success/20 text-success text-xs">✓ スキャン240枚/日</span>
                  <span className="px-2 py-1 rounded bg-success/20 text-success text-xs">✓ 最新モデル先行利用</span>
                  <span className="px-2 py-1 rounded bg-success/20 text-success text-xs">✓ 広告非表示</span>
                </div>
              </div>

              {/* ログアウトボタン */}
              <div className="pt-4 border-t border-gray-700 flex justify-between items-center">
                <button
                  onClick={() => {
                    localStorage.removeItem("auth_token");
                    window.location.reload();
                  }}
                  className="text-sm text-muted hover:text-red-400 transition-colors"
                >
                  ログアウト
                </button>
                <button
                  onClick={handleClose}
                  className="px-4 py-2 rounded-lg font-bold bg-gray-700 hover:bg-gray-600 text-white transition-all"
                >
                  閉じる
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
