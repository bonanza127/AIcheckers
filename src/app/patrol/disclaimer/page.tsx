import Link from "next/link";
import type { Metadata } from "next";
import { Radar } from "lucide-react";

export const metadata: Metadata = {
    title: "免責事項 | AIパトロール",
    description: "AIパトロール（著作権侵害検知・DMCA申請支援）の利用に関する免責事項です。",
};

export default function PatrolDisclaimer() {
    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Header */}
            <header className="site-header border-b border-gray-700">
                <div className="container mx-auto px-4 py-4 flex items-center justify-between">
                    <Link href="/patrol" className="flex items-center gap-1.5 hover:opacity-80">
                        <img src="/logo-transparent.png" alt="AI Checkers" className="w-10 h-10" />
                        <span className="text-xl font-bold text-foreground">AIパトロール</span>
                    </Link>
                    <nav className="text-sm text-muted">
                        <Link href="/patrol" className="hover:text-accent">← パトロールへ戻る</Link>
                    </nav>
                </div>
            </header>

            {/* Main Content */}
            <main className="container mx-auto px-4 py-12 max-w-3xl">
                <div className="text-center mb-10">
                    <Radar className="w-16 h-16 text-cyan-400 mx-auto mb-4" />
                    <h1 className="text-3xl font-bold">免責事項</h1>
                </div>

                <div className="space-y-8 text-muted leading-relaxed">
                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">1. DMCA申請支援について</h2>
                        <p>
                            本サービスは、DMCA（デジタルミレニアム著作権法）に基づく削除申請書の作成を技術的に支援するツールです。
                            本サービスはあくまで申請書作成の補助を行うものであり、実際の申請行為およびその結果（法的紛争、損害賠償請求、相手方からの反論等を含む）について、
                            当サービスは一切の責任を負いません。最終的な申請の判断および送信は、必ず利用者ご自身の責任と意志において行ってください。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">2. 利用資格</h2>
                        <p>
                            本サービスの利用は、対象となる著作物の正当な著作権者ご本人、または著作権者から正式に委任を受けた代理人に限ります。
                            権利を保有しない画像に対する申請、または虚偽の内容を含む申請は固くお断りします。
                            DMCA申請には「偽証罪の罰則の下での宣誓」が含まれており、虚偽の申請を行った場合、法的責任を問われる可能性があります。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">3. 検知精度について</h2>
                        <p>
                            本サービスが提供する類似画像検知機能は、機械学習モデルに基づくものであり、100%の精度を保証するものではありません。
                            誤検知や検知漏れが発生する可能性があることをご了承ください。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">4. アップロード画像</h2>
                        <p>
                            アップロードされた画像は検知処理のみに使用され、サーバーに永続的に保存されることはありません。
                            ただし、違法なコンテンツのアップロードは固くお断りいたします。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">5. サービスの変更・停止/著作権</h2>
                        <p>
                            当サービスは、予告なくサービス内容の変更、または提供を停止することがあります。
                            これによって生じた損害について、当サービスは責任を負いません。
                            また、本サイトのコンテンツ（テキスト、画像、ロゴ等）の著作権は当サービスに帰属します。
                        </p>
                    </section>

                    <section>
                        <h2 className="text-xl font-bold mb-3 text-foreground">6. お問い合わせ</h2>
                        <p>
                            本サービスに関するお問い合わせは、
                            <a href="mailto:contact@aicheckers.net" className="text-cyan-400 hover:underline">contact@aicheckers.net</a>
                            までご連絡ください。
                        </p>
                    </section>
                </div>

                {/* Back Link */}
                <div className="text-center pt-8 mt-8 border-t border-gray-700">
                    <Link
                        href="/patrol"
                        className="inline-block px-6 py-3 bg-cyan-500 text-background font-bold rounded hover:opacity-90 transition"
                    >
                        パトロールへ戻る
                    </Link>
                </div>
            </main>

            {/* Footer */}
            <footer className="site-footer p-4 mt-12">
                <div className="container mx-auto text-center text-xs text-muted">
                    <p><a href="/patrol/disclaimer" className="hover:underline">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="hover:underline">お問い合わせ</a></p>
                </div>
            </footer>
        </div>
    );
}
