import type { Metadata } from "next";
import { Inter, Noto_Sans_JP, Cinzel, Press_Start_2P, DotGothic16, M_PLUS_1_Code } from "next/font/google";
import Script from "next/script";
import "./globals.css";

// Google Analytics 測定ID
const GA_MEASUREMENT_ID = "G-J60G256BKF";

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

const cinzel = Cinzel({
  variable: "--font-cinzel",
  subsets: ["latin"],
  weight: ["400", "500"],
});

const pressStart2P = Press_Start_2P({
  variable: "--font-press-start-2p",
  subsets: ["latin"],
  weight: ["400"],
});

const dotGothic16 = DotGothic16({
  variable: "--font-dot-gothic",
  subsets: ["latin"],
  weight: ["400"],
});

const mPlus1Code = M_PLUS_1_Code({
  variable: "--font-mplus-code",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "AIチェッカー - AI生成画像判定ツール",
  description: "アニメ・イラスト特化のAI画像判定ツール。画像をアップロードするだけで、AIが生成した画像か人間が描いた画像かを高精度で判定します。",
  keywords: ["AIチェッカー", "AIイラストチェッカー", "AI判定", "AI画像検出", "イラスト判定", "アニメ", "生成AI", "画像解析"],
  authors: [{ name: "AI Checkers" }],
  openGraph: {
    title: "AIイラスト判定 | 無料でAI絵を見分けるチェッカー",
    description: "アニメ・イラスト特化のAI画像判定。画像をアップロードするだけで高精度判定。",
    url: "https://aicheckers.net",
    siteName: "AIチェッカー",
    locale: "ja_JP",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "AIイラスト判定 | 無料でAI絵を見分けるチェッカー",
    description: "アニメ・イラスト特化のAI画像判定。画像をアップロードするだけで高精度判定。",
  },
  robots: {
    index: true,
    follow: true,
  },
  metadataBase: new URL("https://aicheckers.net"),
};

// 構造化データ（JSON-LD）
const jsonLd = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  "name": "AI Checkers",
  "alternateName": "AIイラストチェッカー",
  "description": "アニメ・イラスト特化のAI画像判定ツール。画像をアップロードするだけで、AIが生成した画像か人間が描いた画像かを高精度で判定します。",
  "url": "https://aicheckers.net",
  "applicationCategory": "UtilitiesApplication",
  "operatingSystem": "Web",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "JPY"
  },
  "featureList": [
    "AI生成画像の判定",
    "アニメ・イラスト特化",
    "Attention Map可視化",
    "バッチ処理対応"
  ],
  "inLanguage": "ja"
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className={`${inter.variable} ${notoSansJP.variable} ${cinzel.variable} ${pressStart2P.variable} ${dotGothic16.variable} ${mPlus1Code.variable} antialiased`}>
        {/* Google Analytics */}
        <Script
          src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', '${GA_MEASUREMENT_ID}');
          `}
        </Script>
        {children}
      </body>
    </html>
  );
}
