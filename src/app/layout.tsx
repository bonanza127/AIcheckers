import type { Metadata } from "next";
import { Press_Start_2P, Noto_Sans_JP } from "next/font/google";
import "./globals.css";

const pressStart2P = Press_Start_2P({
  variable: "--font-press-start-2p",
  subsets: ["latin"],
  weight: ["400"],
});

const notoSansJP = Noto_Sans_JP({
  variable: "--font-noto-sans-jp",
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
});

export const metadata: Metadata = {
  title: "AI イラストチェッカー | Pixel Art Edition",
  description: "アニメ・イラストがAI生成か人間の作品かを判別するサービス",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className={`${pressStart2P.variable} ${notoSansJP.variable}`}>
        {children}
      </body>
    </html>
  );
}
