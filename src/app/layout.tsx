import type { Metadata } from "next";
import { Noto_Serif_JP } from "next/font/google";
import { Share_Tech_Mono } from "next/font/google";
import "./globals.css";

const notoSerifJP = Noto_Serif_JP({
  variable: "--font-noto-serif-jp",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const shareTechMono = Share_Tech_Mono({
  variable: "--font-share-tech-mono",
  subsets: ["latin"],
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "AI Illustration Checker - アニメイラスト判別",
  description: "アニメ・イラストがAI生成か人間の作品かを判別するサービス",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className={`${notoSerifJP.variable} ${shareTechMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
