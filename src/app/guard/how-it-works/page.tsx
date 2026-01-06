import Link from "next/link";
import type { Metadata } from "next";
import { User, Shield, ChevronLeft } from "lucide-react";

export const metadata: Metadata = {
  title: "よくある質問 | AIイラストガード",
  description: "AIイラストガードに関するよくある質問（FAQ）です。",
};

// FAQ Data
const faqData = [
  {
    q: "AIポイズニングとは？",
    a: "画像に微細な敵対的ノイズを混ぜることにより、AIの学習を妨害する技術です。"
  },
  {
    q: "NightshadeとかGlazeっていうのは聞いたことあるけど、それと何が違うの？",
    a: "原理は同じですが、使用している技術の数と世代が異なります。\n\n2025年8月にLightshadeという技術が登場したことにより、Nightshadeは完全に無効化、その他のGlazeやMist2といったポイズニング技術も併せて突破されてしまうことがわかりました。\n\nそこでラストガードでは、それ以降に登場した複数の最新技術を組み合わせ、単一手法に依存しない多層的な防御を構築しています。"
  },
  {
    q: "実際の効果は？",
    a: "LoRA学習阻害という意味では顕著な効果があります。特に個人で特定の絵師を狙い撃ちするような輩には効果てきめんです。"
  },
  {
    q: "一度使えば効果は永続するの？",
    a: "多少のリサイズには耐性がありますが、何度もリサイズしたり画像を加工したりすると効果が減衰する場合があります。"
  },
  {
    q: "自分の作品はすでにAI学習されてしまっていると思うのだけど…",
    a: "イラストガードにはLoRA学習を破壊するセマンティック（AIの錯覚を引き起こす)攻撃がふくまれているため、毒の混じった画像を学習させればさせるほど、既存の学習結果を悪化させる可能性が高いです。「もう遅い」と感じている方にこそ、お試しいただきたい技術です。"
  },
  {
    q: "でも結局、また新しい技術がでたら、いまのバリアは突破されちゃうんじゃないの？",
    a: "そのとおりです。ゲームにおけるチートとその対策のように、攻撃側と防御側は常にいたちごっこのため、新技術の登場によって、既存のバリアが効かなくなる可能性は大いにあります。\n\nしかし当サイトのMoonknightは、”絶対の防御力”を目指すのではなく、”めちゃめちゃ剥がすのが面倒な納豆のフィルム”のような盾として設計されています。これにより、技術力と根性のある一部のヘンタイ以外には学習を断念される効果があると踏んでいます。"
  }
];

export default function FAQ() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="site-header border-b border-gray-700/50 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/guard" className="flex items-center gap-1.5 hover:opacity-80 transition-opacity">
            <span className="text-xl font-bold bg-gradient-to-r from-accent to-purple-400 bg-clip-text text-transparent">AIイラストガード</span>
          </Link>
          <nav className="text-sm">
            <Link href="/guard" className="flex items-center gap-1 text-muted hover:text-white transition-colors">
              <ChevronLeft className="w-4 h-4" />
              戻る
            </Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8 max-w-2xl">
        <div className="text-center mb-10">
          <h1 className="text-2xl font-bold mb-2">よくある質問</h1>
          <p className="text-muted text-sm">FAQ</p>
        </div>

        <div className="space-y-8">
          {faqData.map((item, index) => (
            <div key={index} className="space-y-4">
              {/* Question (User) */}
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0 mt-1">
                  <User className="w-4 h-4 text-gray-300" />
                </div>
                <div className="bg-gray-800 rounded-2xl rounded-tl-none px-5 py-3 max-w-[85%] shadow-sm">
                  <p className="text-sm md:text-base font-medium text-white">{item.q}</p>
                </div>
              </div>

              {/* Answer (System) */}
              <div className="flex gap-3 justify-end">
                <div className="bg-accent/20 border border-accent/20 rounded-2xl rounded-tr-none px-5 py-3 max-w-[90%] md:max-w-[85%] shadow-sm">
                  {item.a.split('\n').map((line, i) => (
                    <p key={i} className={`text-sm md:text-base text-gray-200 ${line === "" ? "h-2" : ""}`}>
                      {line}
                    </p>
                  ))}
                </div>
                <div className="w-8 h-8 rounded-full bg-accent/20 flex items-center justify-center shrink-0 mt-1 border border-accent/30">
                  <Shield className="w-4 h-4 text-accent" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* CTA Area */}
        <div className="mt-16 text-center">
          <Link
            href="/guard"
            className="inline-flex items-center justify-center px-8 py-3 bg-white text-black font-bold rounded-full hover:bg-gray-200 transition-all shadow-[0_0_20px_rgba(255,255,255,0.3)] hover:shadow-[0_0_30px_rgba(255,255,255,0.5)] active:scale-95"
          >
            さっそく保護する
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="site-footer p-6 mt-auto border-t border-gray-800">
        <div className="container mx-auto text-center text-xs text-muted">
          <p className="flex justify-center gap-4">
            <Link href="/guard/disclaimer" className="hover:text-white transition-colors">利用規約</Link>
            <Link href="/disclaimer" className="hover:text-white transition-colors">免責事項</Link>
            <a href="mailto:contact@aicheckers.net" className="hover:text-white transition-colors">お問い合わせ</a>
          </p>
          <p className="mt-4 text-gray-600">&copy; 2026 AI Checkers</p>
        </div>
      </footer>
    </div>
  );
}
