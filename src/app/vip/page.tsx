import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "VIP会員 | AIチェッカー",
  description: "AIチェッカーのVIP会員登録・ログインページです。無制限スキャンや優先サポートなどの特典をご利用いただけます。",
};

export default function VIPPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="site-header border-b border-gray-700">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-1.5 hover:opacity-80">
            <img src="/logo-transparent.png" alt="AI Checkers" className="w-10 h-10" />
            <span className="text-2xl font-bold text-foreground">AIチェッカー</span>
          </Link>
          <nav className="text-sm text-muted">
            <Link href="/" className="hover:text-accent">← トップへ戻る</Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12 max-w-5xl">
        {/* Hero */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-amber-500/20 to-yellow-500/20 border border-amber-500/50 mb-6">
            <svg className="w-5 h-5 text-amber-400" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
            </svg>
            <span className="font-bold bg-gradient-to-r from-amber-400 to-yellow-300 bg-clip-text text-transparent">
              VIP MEMBERSHIP
            </span>
          </div>
          <h1 className="text-4xl font-extrabold mb-4">
            <span className="bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-400 bg-clip-text text-transparent">
              プレミアム体験を
            </span>
          </h1>
          <p className="text-muted text-lg max-w-2xl mx-auto">
            VIP会員になると、スキャン回数無制限、優先処理、専用サポートなどの特典をご利用いただけます。
          </p>
        </div>

        {/* Benefits */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="card-panel p-6 border-amber-500/30 hover:border-amber-500/50 transition-colors">
            <div className="w-12 h-12 rounded-lg bg-amber-500/20 flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-lg font-bold mb-2 text-foreground">無制限スキャン</h3>
            <p className="text-muted text-sm">1日の制限なし。何枚でもスキャン可能。</p>
          </div>

          <div className="card-panel p-6 border-amber-500/30 hover:border-amber-500/50 transition-colors">
            <div className="w-12 h-12 rounded-lg bg-amber-500/20 flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-bold mb-2 text-foreground">優先処理</h3>
            <p className="text-muted text-sm">混雑時でも優先的に処理。待ち時間を短縮。</p>
          </div>

          <div className="card-panel p-6 border-amber-500/30 hover:border-amber-500/50 transition-colors">
            <div className="w-12 h-12 rounded-lg bg-amber-500/20 flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-bold mb-2 text-foreground">専用サポート</h3>
            <p className="text-muted text-sm">VIP専用の問い合わせ窓口で迅速対応。</p>
          </div>
        </div>

        {/* Pricing */}
        <div className="card-panel p-8 mb-12 border-amber-500/30 bg-gradient-to-br from-amber-500/5 to-yellow-500/5">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-2">料金プラン</h2>
            <p className="text-muted">シンプルな月額制</p>
          </div>
          <div className="flex flex-col md:flex-row items-center justify-center gap-8">
            <div className="text-center">
              <div className="text-5xl font-extrabold bg-gradient-to-r from-amber-400 to-yellow-300 bg-clip-text text-transparent">
                ¥500
              </div>
              <div className="text-muted mt-1">/ 月額</div>
            </div>
            <div className="hidden md:block w-px h-20 bg-gray-700" />
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-success" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span>スキャン回数無制限</span>
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-success" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span>優先キュー処理</span>
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-success" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span>VIP専用サポート</span>
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-success" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span>新機能の先行アクセス</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Auth Section */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Register */}
          <div className="card-panel p-8">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
              </svg>
              新規登録
            </h2>

            <form className="space-y-4">
              <div>
                <label className="block text-sm text-muted mb-1">メールアドレス</label>
                <input
                  type="email"
                  placeholder="your@email.com"
                  className="w-full px-4 py-3 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm text-muted mb-1">パスワード</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-4 py-3 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm text-muted mb-1">パスワード（確認）</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-4 py-3 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                />
              </div>

              {/* Payment Method */}
              <div className="pt-4 border-t border-gray-700">
                <label className="block text-sm text-muted mb-3">お支払い方法</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-3 p-3 rounded-lg border border-gray-700 hover:border-amber-500/50 cursor-pointer transition-colors">
                    <input type="radio" name="payment" value="card" className="text-amber-500" defaultChecked />
                    <svg className="w-5 h-5 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
                    </svg>
                    <span>クレジットカード</span>
                  </label>
                  <label className="flex items-center gap-3 p-3 rounded-lg border border-gray-700 hover:border-amber-500/50 cursor-pointer transition-colors">
                    <input type="radio" name="payment" value="fanbox" className="text-amber-500" />
                    <svg className="w-5 h-5 text-[#F5A623]" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
                    </svg>
                    <div>
                      <span>pixiv FANBOX</span>
                      <span className="text-xs text-muted ml-2">（支援プランから連携）</span>
                    </div>
                  </label>
                </div>
              </div>

              <button
                type="submit"
                className="w-full py-3 rounded-lg font-bold bg-gradient-to-r from-amber-500 to-yellow-500 text-black hover:from-amber-400 hover:to-yellow-400 transition-all shadow-lg shadow-amber-500/25"
              >
                VIP登録する
              </button>
            </form>
          </div>

          {/* Login */}
          <div className="card-panel p-8">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
              </svg>
              ログイン
            </h2>

            <form className="space-y-4">
              <div>
                <label className="block text-sm text-muted mb-1">メールアドレス</label>
                <input
                  type="email"
                  placeholder="your@email.com"
                  className="w-full px-4 py-3 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm text-muted mb-1">パスワード</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-4 py-3 rounded-lg bg-deep-bg border border-gray-700 text-foreground placeholder-gray-500 focus:border-amber-500 focus:outline-none transition-colors"
                />
              </div>

              <button
                type="submit"
                className="w-full py-3 rounded-lg font-bold border-2 border-amber-500 text-amber-400 hover:bg-amber-500/10 transition-all"
              >
                ログイン
              </button>

              <div className="text-center">
                <a href="#" className="text-sm text-muted hover:text-amber-400 transition-colors">
                  パスワードをお忘れですか？
                </a>
              </div>
            </form>

            {/* FANBOX Login */}
            <div className="mt-6 pt-6 border-t border-gray-700">
              <p className="text-sm text-muted text-center mb-4">または</p>
              <button
                type="button"
                className="w-full py-3 rounded-lg font-bold bg-[#F5A623] text-black hover:bg-[#E09620] transition-all flex items-center justify-center gap-2"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
                </svg>
                pixiv FANBOXでログイン
              </button>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="mt-12 text-center">
          <p className="text-muted text-sm">
            ご不明な点がございましたら、
            <a href="mailto:contact@aicheckers.net" className="text-amber-400 hover:underline">contact@aicheckers.net</a>
            までお問い合わせください。
          </p>
        </div>
      </main>

      {/* Footer */}
      <footer className="site-footer p-4 mt-12">
        <div className="container mx-auto text-center text-xs text-muted">
          <p>
            <a href="/disclaimer" className="hover:underline">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. |{" "}
            <a href="mailto:contact@aicheckers.net" className="hover:underline">お問い合わせ</a>
          </p>
        </div>
      </footer>
    </div>
  );
}
