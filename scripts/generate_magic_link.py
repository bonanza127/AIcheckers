#!/usr/bin/env python3
"""
マジックリンク生成ツール

Usage:
  # 開発者用（is_admin=True, is_vip=True）
  python scripts/generate_magic_link.py --type developer --email demo@company.com --name "Company Demo" --days 7

  # VIP会員用（is_admin=False, is_vip=True）
  python scripts/generate_magic_link.py --type vip --email user@example.com --name "VIP User" --days 30
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import jwt

# JWT設定（backend/main.pyと同じ）
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
BASE_URL = "https://api.aicheckers.net"


def generate_magic_link(
    email: str,
    name: str,
    link_type: str = "vip",
    days: int = 30,
) -> str:
    """マジックリンクを生成"""

    if link_type == "developer":
        is_admin = True
        is_vip = True
    elif link_type == "vip":
        is_admin = False
        is_vip = True
    else:
        raise ValueError(f"Unknown link type: {link_type}")

    exp = datetime.now(timezone.utc) + timedelta(days=days)

    payload = {
        "type": "magic_link",
        "email": email,
        "name": name,
        "is_admin": is_admin,
        "is_vip": is_vip,
        "exp": exp,
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    magic_link = f"{BASE_URL}/auth/magic/{token}"

    return magic_link, exp, is_admin, is_vip


def main():
    parser = argparse.ArgumentParser(description="マジックリンク生成ツール")
    parser.add_argument("--type", choices=["developer", "vip"], required=True,
                        help="リンク種別: developer (管理者+VIP) / vip (VIPのみ)")
    parser.add_argument("--email", required=True, help="メールアドレス")
    parser.add_argument("--name", required=True, help="表示名")
    parser.add_argument("--days", type=int, default=30, help="有効期限（日数）")

    args = parser.parse_args()

    link, exp, is_admin, is_vip = generate_magic_link(
        email=args.email,
        name=args.name,
        link_type=args.type,
        days=args.days,
    )

    print()
    print("=" * 60)
    print("  マジックリンク生成完了")
    print("=" * 60)
    print(f"  種別:     {args.type.upper()}")
    print(f"  名前:     {args.name}")
    print(f"  メール:   {args.email}")
    print(f"  is_admin: {is_admin}")
    print(f"  is_vip:   {is_vip}")
    print(f"  有効期限: {exp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("-" * 60)
    print()
    print(link)
    print()


if __name__ == "__main__":
    main()
