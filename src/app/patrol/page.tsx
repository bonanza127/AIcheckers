"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Shield, Search, AlertTriangle, CheckCircle, ExternalLink, Settings, Eye, Zap, ChevronRight, Activity, Globe, Lock, X, Mail, Copy, Check, RotateCcw } from "lucide-react";
import HamburgerMenu from "@/components/HamburgerMenu";
import VipModal from "@/components/VipModal";

// Types
type Alert = {
    id: string;
    thumbnail: string;
    sourceUrl: string;
    sourceSite: string;
    similarity: number; // 0-100
    timestamp: string;
    status: "pending" | "ignored" | "reported";
    // TrustMark情報（Guard保護時に埋め込まれた情報）
    watermarkHash?: string;       // 61bitの透かしハッシュ
    protectionTimestamp?: string; // 保護した日時（UTC）
    protectedBy?: string;         // 保護したユーザーID（匿名化済み）
    dinov3Similarity?: number;    // DINOv3類似度（0-1）
};

type Stat = {
    label: string;
    value: string;
    icon: React.ElementType;
    color: string;
};

// SVGプレースホルダー生成（軽量、外部リソース不要）
const generatePlaceholderSvg = (index: number): string => {
    const hues = [200, 260, 320, 160, 30, 0];
    const hue = hues[index % hues.length];
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect fill="hsl(${hue},30%,15%)" width="400" height="300"/><text x="200" y="140" text-anchor="middle" fill="hsl(${hue},60%,60%)" font-size="14" font-family="monospace">Detected Image</text><text x="200" y="170" text-anchor="middle" fill="hsl(${hue},40%,40%)" font-size="12" font-family="monospace">#${String(index + 1).padStart(3, '0')}</text></svg>`;
    return `data:image/svg+xml,${encodeURIComponent(svg)}`;
};

// Mock Data Generator
const generateMockAlerts = (count: number): Alert[] => {
    return Array.from({ length: count }).map((_, index) => {
        // Hydrationエラーを防ぐため、ランダムではなくインデックスに基づいた決定論的な値を生成
        const scoreBase = [91.4, 96.8, 87.8, 92.8, 89.4, 94.8];
        const matchScore = scoreBase[index % scoreBase.length].toFixed(1);

        // Mock TrustMark data
        // 日付も固定化する
        const mockDate = new Date("2024-01-08T14:00:00.000Z");
        mockDate.setDate(mockDate.getDate() - index);
        const mockTimestamp = mockDate.toISOString();

        // ハッシュもインデックスベースで生成
        const mockHash = Array.from({ length: 61 }, (_, i) => ((index + i) % 2).toString()).join('');

        return {
            id: `alert-${index + 1}`,
            thumbnail: generatePlaceholderSvg(index),
            sourceUrl: `https://pirate-site-${String.fromCharCode(65 + (index % 5))}.com/gallery/user/${12345 + index}`,
            sourceSite: `Pirate-Site-${String.fromCharCode(65 + (index % 5))}.com`,
            similarity: parseFloat(matchScore),
            timestamp: mockTimestamp, // 使用されていないようだが型にあるので
            status: "pending",
            // TrustMark情報
            watermarkHash: mockHash,
            protectionTimestamp: mockTimestamp,
            protectedBy: `user_${String(1000 + index).slice(-4)}`,
            dinov3Similarity: 0.95 + (index % 5) * 0.01,
        };
    });
};

const MOCK_ALERTS = generateMockAlerts(26);

// DMCA Template with TrustMark Evidence
const getDmcaTemplate = (alert: Alert) => {
    const hasWatermark = alert.watermarkHash && alert.protectionTimestamp;
    const protectionDate = alert.protectionTimestamp
        ? new Date(alert.protectionTimestamp).toLocaleDateString('ja-JP', { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })
        : '[保護日時不明]';

    const evidenceSection = hasWatermark ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▼ DIGITAL WATERMARK EVIDENCE (電子透かし証拠)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This image contains an invisible digital watermark embedded using AIイラストガード (AI Illustration Guard) that proves my ownership.
この画像には、私の所有権を証明するAIイラストガードによる不可視の電子透かしが埋め込まれています。

■ Protection Date (保護日時): ${protectionDate}
■ Watermark Hash (透かしハッシュ): ${alert.watermarkHash}
■ Visual Similarity (視覚類似度): ${alert.similarity}%
■ AI Embedding Match (AI埋込一致度): ${alert.dinov3Similarity ? (alert.dinov3Similarity * 100).toFixed(1) : 'N/A'}%

This watermark can be independently verified using the AIチェッカー Patrol system.
この透かしはAIチェッカーのPatrolシステムで独立して検証可能です。
Verification URL: https://aicheckers.net/patrol/verify?hash=${alert.watermarkHash?.slice(0, 16)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━` : '';

    return {
        to: `dmca@${alert.sourceSite.toLowerCase()}`,
        subject: `DMCA Takedown Notice - ${alert.sourceSite} [Digital Watermark Evidence Attached]`,
        body: `To The Administrator of ${alert.sourceSite},

I am the copyright owner of the artwork being displayed on your website without my permission. I hereby request specifically that you remove the following infringing material from your website and servers.

▼ INFRINGING CONTENT (侵害コンテンツ)
URL: ${alert.sourceUrl}
Discovery Date: ${alert.timestamp}
${evidenceSection}${hasWatermark ? `
▼ ADDITIONAL REFERENCES (補足情報・任意)
※ The digital watermark above serves as the primary proof of ownership.
※ 上記の電子透かしが所有権の主要な証拠です。

Original Post URL (if available): [あれば記入]
Social Media Profile: [SNSプロフィールURL]` : `
▼ MY ORIGINAL WORK (私のオリジナル作品)
Original Post URL: [あなたの元投稿のURL]
Creation Date: [作成日]`}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▼ LEGAL DECLARATION (法的宣誓)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I have a good faith belief that use of the material in the manner complained of is not authorized by the copyright owner, its agent, or the law.

I swear, under penalty of perjury, that the information in the notification is accurate, and that I am the copyright owner or am authorized to act on behalf of the owner of an exclusive right that is allegedly infringed.

私は、問題となっている方法での素材の使用が著作権者、その代理人、または法律により許可されていないと誠実に信じています。

私は、偽証罪の罰則の下、この通知の情報が正確であり、私が著作権者であるか、侵害されているとされる排他的権利の所有者を代表して行動する権限を与えられていることを誓います。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sincerely / 敬具,
[Your Name / お名前]
[Your Contact Information / 連絡先]
[Your Address (Required for DMCA) / 住所（DMCA必須）]`
    };
};


export default function PatrolPage() {
    const [isActive, setIsActive] = useState(true);
    const [alerts, setAlerts] = useState<Alert[]>(MOCK_ALERTS);
    const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
    const [isVipModalOpen, setIsVipModalOpen] = useState(false);

    // Flip State
    const [isFlipped, setIsFlipped] = useState(false);
    const [dmcaData, setDmcaData] = useState<{ to: string, subject: string, body: string } | null>(null);

    const [backendOnline, setBackendOnline] = useState<boolean | null>(true);
    const [copiedField, setCopiedField] = useState<string | null>(null);
    const [authUser, setAuthUser] = useState<{ name: string, email: string, token: string, isVip: boolean, isAdmin?: boolean } | null>(null);

    // Reset flip when selection changes
    useEffect(() => {
        setIsFlipped(false);
    }, [selectedAlert?.id]);

    const stats: Stat[] = [
        { label: "監視中の画像", value: "248", icon: Lock, color: "text-cyan-400" },
        { label: "監視サイト数", value: "26", icon: Globe, color: "text-purple-400" },
        { label: "高リスク検知", value: String(alerts.filter(a => a.status === "pending").length), icon: AlertTriangle, color: "text-red-400" },
    ];

    const handleReportClick = (alert: Alert) => {
        setDmcaData(getDmcaTemplate(alert));
        setIsFlipped(true);
    };

    const handleAction = (id: string, action: "ignore" | "report_done") => {
        if (action === "report_done") {
            setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: "reported" } : a));
            setIsFlipped(false);
            setSelectedAlert(null);
        } else {
            setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: "ignored" } : a));
            setSelectedAlert(null);
        }
    };

    const handleCopy = (text: string, field: string) => {
        navigator.clipboard.writeText(text);
        setCopiedField(field);
        setTimeout(() => setCopiedField(null), 2000);
    };

    return (
        <div className="min-h-screen flex flex-col font-sans selection:bg-cyan-900 selection:text-cyan-100">
            {/* Header */}
            <header className="site-header sticky top-0 z-40 p-4">
                <div className="container mx-auto flex justify-between items-center">
                    <div className="flex items-center gap-0">
                        <HamburgerMenu variant="patrol" />
                        <img src="/logo-transparent.png" alt="AI Checkers" className="w-14 h-14" />
                        <h2 className="text-lg md:text-2xl font-bold tracking-tight whitespace-nowrap flex items-center text-white">
                            AIパトロール
                            <span className="hidden md:inline text-sm font-light text-muted mx-2 opacity-50">//</span>
                            <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-900/50">開発中</span>
                        </h2>
                    </div>

                    <div className="flex items-center gap-4 text-xs">
                        {/* Status Indicators */}
                        <div className="flex items-center gap-1.5">
                            <span className={`w-1.5 h-1.5 rounded-full ${backendOnline === null ? "bg-gray-500 animate-pulse" : backendOnline ? "bg-green-500" : "bg-red-500"}`} />
                            <span className="hidden md:inline text-gray-500">Server Status:</span>
                            <span className={backendOnline ? "text-green-500" : "text-red-500"}>
                                {backendOnline ? "Online" : "Offline"}
                            </span>
                        </div>

                        <button
                            onClick={() => setIsVipModalOpen(true)}
                            className="group relative px-4 py-1.5 font-[family-name:var(--font-cinzel)] text-[10px] font-medium tracking-[0.2em] transition-all duration-500 bg-zinc-900/50 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] rounded-sm text-zinc-500 border border-zinc-700/50 hover:text-zinc-300 hover:border-zinc-600"
                        >
                            VIP
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="flex-grow container mx-auto px-4 py-6 relative z-10 grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-140px)] min-h-0">

                {/* Left: Feed (Scrollable, 2/3 width) */}
                <div className="lg:col-span-2 flex flex-col min-h-0 card-panel relative h-full overflow-hidden border-zinc-800 bg-black/40">
                    <div className="p-5 border-b border-white/5 bg-black/20 backdrop-blur sticky top-0 z-10 flex flex-col md:flex-row md:items-center gap-2 md:gap-6">
                        <h3 className="panel-header mb-0 border-none pb-0 whitespace-nowrap">Alert Feed</h3>
                        <div className="text-xs text-zinc-400 leading-relaxed">
                            <p>AIイラストガードで記録したハッシュをもとに、無断転載を追跡。</p>
                            <p className="opacity-80">今度はAIがあなたの味方として、ネット上をパトロール。発見から削除申請までお手伝いします。</p>
                        </div>
                    </div>

                    <div className="overflow-y-auto p-4 space-y-3 custom-scrollbar flex-grow bg-transparent">
                        {alerts.filter(a => a.status === "pending").length === 0 ? (
                            <div className="card-panel p-12 text-center flex flex-col items-center justify-center opacity-70">
                                <CheckCircle className="text-green-500 mb-4" size={32} />
                                <p className="text-zinc-400 font-medium">All clear. No pending alerts.</p>
                            </div>
                        ) : (
                            alerts.filter(a => a.status === "pending").map((alert) => (
                                <div
                                    key={alert.id}
                                    onClick={() => setSelectedAlert(alert)}
                                    className={`card-panel p-3 flex gap-4 cursor-pointer group transition-all duration-200 hover:translate-x-1 hover:border-zinc-600 ${selectedAlert?.id === alert.id ? "border-cyan-500/50 bg-cyan-900/10" : "border-zinc-800"}`}
                                >
                                    {/* Thumbnail Preview */}
                                    <div className="relative w-24 h-24 rounded-lg overflow-hidden bg-black flex-shrink-0 border border-zinc-800 group-hover:border-zinc-600 transition-colors">
                                        <img src={alert.thumbnail} loading="lazy" alt="Alert thumbnail" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                                        <span className="absolute bottom-1 right-1 text-[10px] font-mono bg-black/50 backdrop-blur px-1 rounded text-white">{alert.similarity}%</span>
                                    </div>

                                    <div className="flex-1 min-w-0 flex flex-col justify-center">
                                        <div className="flex justify-between items-start mb-1">
                                            <div className="flex items-center gap-2">
                                                <span className="px-1.5 py-0.5 rounded text-[0.6rem] font-bold uppercase tracking-wide bg-red-500/10 text-red-400 border border-red-500/20">
                                                    High Risk
                                                </span>
                                                <span className="text-[0.65rem] text-zinc-500 font-mono">{alert.timestamp}</span>
                                            </div>
                                        </div>

                                        <h4 className="font-bold text-zinc-100 text-sm truncate mb-1 group-hover:text-cyan-400 transition-colors">{alert.sourceSite}</h4>
                                        <p className="text-xs text-zinc-500 truncate font-mono opacity-80">{alert.sourceUrl}</p>
                                    </div>

                                    <div className="flex items-center justify-center px-2">
                                        <ChevronRight className="text-zinc-700 group-hover:text-cyan-500 transition-colors" size={16} />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Right Column: Stats & Detail (1/3 width, Scrollable) */}
                <div className="lg:col-span-1 flex flex-col gap-4 min-h-0 overflow-y-auto pb-4 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">

                    {/* 1. Compact Stats Panel */}
                    <div className="card-panel p-3 relative overflow-hidden group flex-shrink-0">
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-900/5 via-transparent to-purple-900/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />
                        <div className="flex items-center justify-between relative z-10">
                            {/* Stats Grid */}
                            <div className="flex gap-6">
                                {stats.map((stat, i) => (
                                    <div key={i} className="text-left">
                                        <div className="flex items-center gap-1.5 text-zinc-500 text-[9px] uppercase tracking-widest font-bold mb-0.5">
                                            <stat.icon size={10} />
                                            {stat.label}
                                        </div>
                                        <div className={`text-xl font-bold font-mono tracking-tight ${stat.color}`}>
                                            {stat.value}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <button
                                onClick={() => setIsActive(!isActive)}
                                className={`flex-shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 ${isActive ? "bg-cyan-600" : "bg-zinc-800 border border-zinc-700"}`}
                            >
                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-md transition-transform duration-300 ${isActive ? "translate-x-6" : "translate-x-1"}`} />
                            </button>
                        </div>
                    </div>

                    {/* 2. Flippable Detail View */}
                    {selectedAlert ? (
                        <div className="relative w-full flex-grow [perspective:1000px]">
                            <div className={`relative w-full h-full transition-transform duration-700 [transform-style:preserve-3d] ${isFlipped ? "[transform:rotateY(180deg)]" : ""}`}>

                                {/* FRONT: Detail View */}
                                <div className="absolute inset-0 w-full h-full [backface-visibility:hidden]">
                                    <div className="card-panel h-full overflow-hidden flex flex-col">
                                        <div className="flex items-center justify-between p-3 border-b border-zinc-800 bg-[#161B22]">
                                            <h3 className="panel-header mb-0 border-none pb-0 text-xs">Detail View</h3>
                                            <div className="text-[10px] text-zinc-500 font-mono">ID: {selectedAlert.id}</div>
                                        </div>

                                        {/* Image Preview - Theater Mode */}
                                        <div className="relative flex-grow bg-black/50 overflow-hidden group min-h-[50%]">
                                            <div
                                                className="absolute inset-0 bg-cover bg-center opacity-30 blur-xl scale-110"
                                                style={{ backgroundImage: `url(${selectedAlert.thumbnail})` }}
                                            />
                                            <div className="absolute inset-0 flex items-center justify-center p-6">
                                                <img
                                                    src={selectedAlert.thumbnail}
                                                    alt="Alert preview"
                                                    className="max-w-full max-h-full object-contain shadow-2xl drop-shadow-[0_0_15px_rgba(0,0,0,0.5)] rounded-sm border border-white/10"
                                                />
                                            </div>
                                            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 via-black/50 to-transparent">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <Activity size={14} className="text-cyan-400" />
                                                    <span className="text-cyan-400 font-mono font-bold text-lg tracking-widest">{selectedAlert.similarity}% MATCH</span>
                                                </div>
                                                <a href={selectedAlert.sourceUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-zinc-300 hover:text-white flex items-center gap-1 hover:underline truncate">
                                                    {selectedAlert.sourceUrl} <ExternalLink size={10} />
                                                </a>
                                            </div>
                                        </div>

                                        {/* Stacked Actions */}
                                        <div className="p-4 space-y-3 bg-[#161B22] border-t border-zinc-800">
                                            <button
                                                onClick={() => handleReportClick(selectedAlert)}
                                                className="primary-button flex items-center justify-center gap-2 py-4 w-full text-sm shadow-lg shadow-purple-500/10"
                                            >
                                                <Shield size={16} />
                                                DMCA申請を作成
                                            </button>
                                            <button
                                                onClick={() => handleAction(selectedAlert.id, "ignore")}
                                                className="w-full py-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 font-medium rounded-lg border border-zinc-700 hover:border-zinc-600 transition-all text-sm hover:text-white"
                                            >
                                                誤検知として無視
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {/* BACK: DMCA Form */}
                                <div className="absolute inset-0 w-full h-full [backface-visibility:hidden] [transform:rotateY(180deg)]">
                                    <div className="card-panel h-full overflow-hidden flex flex-col bg-[#0C1117] border-cyan-500/30 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
                                        <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-[#161B22]">
                                            <h3 className="text-sm font-bold text-white flex items-center gap-2">
                                                <Mail className="text-cyan-400" size={16} />
                                                DMCA Template
                                            </h3>
                                            <button onClick={() => setIsFlipped(false)} className="text-zinc-500 hover:text-white transition-colors bg-zinc-800 p-1 rounded-full">
                                                <RotateCcw size={14} />
                                            </button>
                                        </div>

                                        {/* Reminder Banner */}
                                        <div className="bg-amber-950/40 border-b border-amber-900/50 p-3 flex items-start gap-3 backdrop-blur-sm relative z-10 animate-in fade-in slide-in-from-top-2 duration-500">
                                            <AlertTriangle className="text-amber-500 shrink-0 mt-0.5" size={16} />
                                            <div className="text-xs text-amber-200/90 leading-relaxed">
                                                <p className="font-bold mb-1 text-amber-400">送信前の書き換えチェック</p>
                                                <ul className="list-disc list-inside opacity-90 space-y-1">
                                                    <li>（もしあれば）補足情報のOriginal Post URLとSNSのURLを入力</li>
                                                    <li>末尾の <span className="text-white font-mono bg-white/10 px-1 rounded mx-1">[名前]</span> <span className="text-white font-mono bg-white/10 px-1 rounded mx-1">[連絡先]</span> <span className="text-white font-mono bg-white/10 px-1 rounded mx-1">[住所]</span> は自分のものに書き換え</li>
                                                </ul>
                                            </div>
                                        </div>

                                        <div className="flex-grow overflow-y-auto p-4 space-y-4 custom-scrollbar bg-black/20">
                                            {dmcaData && [
                                                { label: "To", key: "to", value: dmcaData.to, multiline: false },
                                                { label: "Subject", key: "subject", value: dmcaData.subject, multiline: false },
                                                { label: "Body", key: "body", value: dmcaData.body, multiline: true },
                                            ].map((field) => (
                                                <div key={field.key} className="space-y-1">
                                                    <div className="flex justify-between items-center">
                                                        <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">{field.label}</label>
                                                        <button
                                                            onClick={() => handleCopy(field.value as string, field.key)}
                                                            className={`text-[10px] flex items-center gap-1 px-2 py-0.5 rounded transition-colors ${copiedField === field.key ? "bg-green-500 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"}`}
                                                        >
                                                            {copiedField === field.key ? <Check size={10} /> : <Copy size={10} />}
                                                            {copiedField === field.key ? "Copied" : "Copy"}
                                                        </button>
                                                    </div>
                                                    {field.multiline ? (
                                                        <textarea
                                                            readOnly
                                                            value={field.value as string}
                                                            className="w-full h-80 bg-black/60 border border-zinc-700/50 rounded p-3 text-xs text-zinc-300 focus:outline-none focus:border-cyan-500/50 font-mono resize-none leading-relaxed custom-scrollbar"
                                                        />
                                                    ) : (
                                                        <input
                                                            readOnly
                                                            type="text"
                                                            value={field.value as string}
                                                            className="w-full bg-black/60 border border-zinc-700/50 rounded p-2 text-xs text-zinc-300 focus:outline-none focus:border-cyan-500/50 font-mono"
                                                        />
                                                    )}
                                                </div>
                                            ))}
                                        </div>

                                        <div className="p-4 border-t border-zinc-800 bg-[#161B22]">
                                            <button
                                                onClick={() => handleAction(selectedAlert.id, "report_done")}
                                                className="w-full py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-bold rounded shadow-lg hover:shadow-cyan-500/20 hover:-translate-y-0.5 transition-all text-sm flex items-center justify-center gap-2"
                                            >
                                                <Check size={16} />
                                                送信済みとして完了
                                            </button>
                                        </div>
                                    </div>
                                </div>

                            </div>
                        </div>
                    ) : (
                        <div className="card-panel p-8 text-center flex flex-col items-center justify-center flex-grow border-dashed border-2 border-zinc-800 bg-transparent opacity-50">
                            <Search className="text-zinc-700 mb-4" size={32} />
                            <p className="text-zinc-500 text-sm">Select an alert to view details</p>
                        </div>
                    )}

                </div>
            </main>

            {/* Footer */}
            <footer className="site-footer p-4 border-t border-white/5 bg-black/40 backdrop-blur-sm relative z-10 flex-shrink-0">
                <div className="container mx-auto text-center text-xs text-zinc-500">
                    <p><Link href="/patrol/disclaimer" className="hover:text-zinc-300 transition-colors">免責事項</Link> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="hover:text-zinc-300 transition-colors">お問い合わせ</a></p>
                </div>
            </footer>

            <VipModal isOpen={isVipModalOpen} onClose={() => setIsVipModalOpen(false)} authUser={authUser} />
        </div>
    );
}
