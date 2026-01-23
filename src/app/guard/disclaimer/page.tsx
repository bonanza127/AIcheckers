import Link from "next/link";
import type { Metadata } from "next";
import { Shield } from "lucide-react";

export const metadata: Metadata = {
    title: "免責事項 | AIイラストガード",
    description: "AIイラストガード（Moonknight）の利用に関する免責事項です。",
};

export default function GuardDisclaimer() {
    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Header */}
            <header className="site-header border-b border-gray-700">
                <div className="container mx-auto px-4 py-4 flex items-center justify-between">
                    <Link href="/guard" className="flex items-center gap-1.5 hover:opacity-80">
                        <img src="/logo-transparent.png" alt="AI Checkers" className="w-10 h-10" />
                        <span className="text-xl font-bold text-foreground">AIイラストガード</span>
                    </Link>
                    <nav className="text-sm text-muted">
                        <Link href="/guard" className="hover:text-accent">← ガードへ戻る</Link>
                    </nav>
                </div>
            </header>

            {/* Main Content */}
            <main className="container mx-auto px-4 py-12 max-w-3xl">
                <div className="text-center mb-10">
                    <Shield className="w-16 h-16 text-accent mx-auto mb-4" />
                    <h1 className="text-3xl font-bold">免責事項</h1>

                </div>

                <div className="space-y-8 text-muted leading-relaxed">
                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">1. 保護機能について</h2>
                        <p>
                            本サービスが提供する画像保護機能は、機械学習モデルに対する妨害を目的としたものであり、
                            100%の保護を保証するものではありません。本サービスの使用、または期待した効果が得られなかったことによって生じたいかなる損害についても、当サービスは責任を負いかねます。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">2. 画質への影響</h2>
                        <p>
                            保護強度に応じて、画像に微細なノイズ等の視覚的変化が生じる場合があります。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">3. 注意事項</h2>
                        <p>
                            画像のリサイズや加工等により、保護機能が減衰する場合があります。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">4. 利用目的</h2>
                        <p>
                            本サービスは、画像をAI学習から保護するためのツールとして提供しています。
                            本サービスを用いた誹謗中傷、権利侵害の主張、その他の悪意ある目的での使用を禁止します。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">5. アップロード画像</h2>
                        <p>
                            アップロードされた画像は保護処理のみに使用され、サーバーに保存されることはありません。
                            ただし、違法なコンテンツのアップロードは固くお断りいたします。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">6. サービスの変更・停止/著作権</h2>
                        <p>
                            当サービスは、予告なくサービス内容の変更、または提供を停止することがあります。
                            これによって生じた損害について、当サービスは責任を負いません。
                            また、本サイトのコンテンツ（テキスト、画像、ロゴ等）の著作権は当サービスに帰属します。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">7. お問い合わせ</h2>
                        <p>
                            本サービスに関するお問い合わせは、
                            <a href="mailto:contact@aicheckers.net" className="text-accent hover:underline">contact@aicheckers.net</a>
                            までご連絡ください。
                        </p>
                    </section>
                </div>

                {/* Back Link */}
                <div className="text-center pt-8 mt-8 border-t border-gray-700">
                    <Link
                        href="/guard"
                        className="inline-block px-6 py-3 bg-accent text-background font-bold rounded hover:opacity-90 transition"
                    >
                        ガードへ戻る
                    </Link>
                </div>
            </main>

            {/* Footer */}
            <footer className="site-footer p-4 mt-12">
                <div className="container mx-auto text-center text-xs text-muted">
                    <p><a href="/guard/disclaimer" className="hover:underline">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="hover:underline">お問い合わせ</a></p>
                </div>
            </footer>
        </div>
    );
}
