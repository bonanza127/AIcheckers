"use client";

import { useState, useRef, useEffect } from "react";
import { Menu, Shield, Search } from "lucide-react";
import Link from "next/link";

type HamburgerMenuProps = {
  variant?: "checker" | "guard";
};

export default function HamburgerMenu({ variant = "checker" }: HamburgerMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // 外側クリックで閉じる
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  // チェッカーページではガードへ、ガードページではチェッカーへ
  const menuItem = variant === "checker" ? {
    href: "/guard",
    icon: Shield,
    title: "AIイラストガード",
    description: "無断学習から作品を保護"
  } : {
    href: "/",
    icon: Search,
    title: "AIイラストチェッカー",
    description: "AI生成画像を判別"
  };

  const Icon = menuItem.icon;

  return (
    <div className="relative" ref={menuRef}>
      {/* ハンバーガーボタン */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`p-2 rounded-lg transition-all duration-300 ${
          isOpen
            ? "bg-accent/20 text-accent border border-accent/40"
            : "text-zinc-400 border border-transparent hover:text-zinc-200 hover:bg-zinc-800/50 hover:border-zinc-700"
        }`}
        aria-label="メニューを開く"
      >
        <Menu className="w-5 h-5" strokeWidth={1.5} />
      </button>

      {/* ドロップダウン */}
      {isOpen && (
        <div className="absolute left-0 top-full mt-2 w-56 bg-zinc-900/95 backdrop-blur-sm border border-zinc-700/50 rounded-lg shadow-xl shadow-black/30 overflow-hidden z-50">
          <div className="py-1">
            <Link
              href={menuItem.href}
              onClick={() => setIsOpen(false)}
              className="flex items-center gap-3 px-4 py-3 text-sm text-zinc-300 hover:bg-accent/10 hover:text-accent transition-colors"
            >
              <Icon className="w-4 h-4" />
              <div>
                <div className="font-medium">{menuItem.title}</div>
                <div className="text-xs text-zinc-500">{menuItem.description}</div>
              </div>
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
