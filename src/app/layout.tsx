import type { Metadata } from "next";
import { Inter, Noto_Sans_JP } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

const notoSansJP = Noto_Sans_JP({
  variable: "--font-noto-sans-jp",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "AI Checkers - AI生成画像判定ツール",
  description: "アニメ・イラスト特化のAI画像判定ツール。画像をアップロードするだけで、AIが生成した画像か人間が描いた画像かを高精度で判定します。",
  keywords: ["AI判定", "AI画像検出", "イラスト判定", "アニメ", "生成AI", "画像解析"],
  authors: [{ name: "AI Checkers" }],
  openGraph: {
    title: "AI Checkers - AI生成画像判定ツール",
    description: "アニメ・イラスト特化のAI画像判定。画像をアップロードするだけで高精度判定。",
    url: "https://aicheckers.net",
    siteName: "AI Checkers",
    locale: "ja_JP",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "AI Checkers - AI生成画像判定ツール",
    description: "アニメ・イラスト特化のAI画像判定。画像をアップロードするだけで高精度判定。",
  },
  robots: {
    index: true,
    follow: true,
  },
  metadataBase: new URL("https://aicheckers.net"),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className={`${inter.variable} ${notoSansJP.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
