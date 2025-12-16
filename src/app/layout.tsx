import type { Metadata } from "next";
import { Noto_Sans_JP } from "next/font/google";
import { Share_Tech_Mono } from "next/font/google";
import "./globals.css";

const notoSansJP = Noto_Sans_JP({
  variable: "--font-noto-sans-jp",
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
});

const shareTechMono = Share_Tech_Mono({
  variable: "--font-share-tech-mono",
  subsets: ["latin"],
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "SYSTEM // ARCHIVE:AI.RDT - AI Image Detector",
  description: "アニメ・イラストがAI生成か人間の作品かを判別するサービス",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body
        className={`${notoSansJP.variable} ${shareTechMono.variable}`}
      >
        {children}
      </body>
    </html>
  );
}
