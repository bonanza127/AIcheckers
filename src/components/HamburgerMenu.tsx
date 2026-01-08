"use client";

import { useState, useRef, useEffect } from "react";
import { Menu, Shield, Search, Eye } from "lucide-react";
import Link from "next/link";

type HamburgerMenuProps = {
  variant?: "checker" | "guard" | "patrol";
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

  const items = [
    {
      id: "checker",
      href: "/",
      icon: Search,
      title: "AIイラストチェッカー",
      description: "AI生成画像を判別",
      visible: variant !== "checker"
    },
    {
      id: "guard",
      href: "/guard",
      icon: Shield,
      title: "AIイラストガード",
      description: "無断学習から作品を保護",
      visible: variant !== "guard"
    },
    {
      id: "patrol",
      href: "/patrol",
      icon: Eye,
      title: "無断転載追跡サービス",
      description: "追跡から削除申請のお手伝いまで",
      tag: "仮",
      visible: variant !== "patrol"
    }
  ];

  // variant = "patrol" のときは Checker -> Guard の順になるように (itemsの定義順でOK)
  // variant = "checker" のときは Guard -> Patrol
  // variant = "guard" のときは Checker -> Patrol

  // しかしChecker/Guardページでは相手を先に見せたいかもしれないが、一旦定義順で表示し、自分自身を除外するロジックにする。
  // 要望: "一番上にチェッカー、その次にガードが表示されるように" (Patrolページの場合) -> itemsの順序通り(Checker, Guard, Patrol)でfilterすればOK。

  return (
    <div className="relative" ref={menuRef}>
      {/* ハンバーガーボタン */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`p-2 rounded-lg transition-all duration-300 ${isOpen
          ? "bg-accent/20 text-accent border border-accent/40"
          : "text-zinc-400 border border-transparent hover:text-zinc-200 hover:bg-zinc-800/50 hover:border-zinc-700"
          }`}
        aria-label="メニューを開く"
        id="hamburger-button"
      >
        <Menu className="w-5 h-5" strokeWidth={1.5} />
      </button>

      {/* ドロップダウン */}
      {isOpen && (
        <div className="absolute left-0 top-full mt-2 w-64 bg-zinc-900/95 backdrop-blur-sm border border-zinc-700/50 rounded-lg shadow-xl shadow-black/30 overflow-hidden z-50 animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="py-1">
            {items.filter(i => i.visible).map((item, index) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.id}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 text-sm text-zinc-300 hover:text-white transition-colors border-b border-zinc-800 last:border-0 ${item.id === "patrol" ? "hover:bg-cyan-900/10 hover:text-cyan-400" : "hover:bg-accent/10 hover:text-accent"
                    }`}
                >
                  <Icon className="w-4 h-4" />
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      {item.title}
                      {item.tag && (
                        <span className="text-[10px] bg-cyan-900/30 text-cyan-400 px-1 rounded border border-cyan-800">{item.tag}</span>
                      )}
                    </div>
                    <div className="text-xs text-zinc-500">{item.description}</div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
