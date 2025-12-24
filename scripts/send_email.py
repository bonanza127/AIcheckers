#!/usr/bin/env python3
"""
Resend経由でメール送信
Usage: python scripts/send_email.py [--send]
  --send なしでプレビュー、ありで実際に送信
"""
import argparse
import resend

# API Key
resend.api_key = "re_2PH8P2MW_2FFoKCL2F8r46b7ZJs1kD4hd"

# メール設定
FROM_EMAIL = "石川風水 <contact@aicheckers.net>"
TO_EMAIL = "corp@eisys.co.jp"  # DLsite運営会社
SUBJECT = "【ご提案】AI生成画像検出ツールのご紹介"

BODY = """\
株式会社エイシス
DLsite 運営ご担当者様

突然のご連絡失礼いたします。
画像解析アルゴリズムの開発および、生成画像検証プラットフォーム「AIチェッカー」を運営しております、石川風水と申します。

生成AIの普及に伴い、AI生成画像を手描きと偽って投稿・販売する事例が増加しており、プラットフォーム運営においても不要なリスクが年々高まっている状況かと存じます。

このような課題に対する解決策として、アニメ調・二次元イラストに特化したAI生成画像チェッカーを開発いたしました。

本ツールでは、現在CivitAI等で主流となっているモデルに対して高い検出精度を確認しており、
具体的には Pony Diffusion 系で約99%、Illustrious 系で約98%の検出率を記録しております。

これらは学習済みデータに基づく数値ではありますが、実際にそれらをベースとした派生チェックポイントや、LoRAを適用して生成された画像に対して、どの程度の精度が得られるかをご確認いただけるよう、Webサイト上にてデモ版を公開しております。
https://www.aicheckers.net

また、検証用途として、エイシス様専用のログインリンクをご用意いたしました。
こちらのリンクからアクセスいただくことで、回数制限なく画像検証をお試しいただけます。

検証用リンク：https://aicheckers.net/trial1
有効期限：2026年1月15日まで

年末年始でお忙しい時期かと存じますので、もし十分にご検証いただけなかった場合には、改めてリンクを発行いたします。どうぞご遠慮なくお申し付けください。

本技術の精度や実用性をご評価いただけました際には、APIサービスとしての正式なご検討を賜れましたら幸いです。

本サービスは、以下の形での提供を想定しております。

料金：1画像あたり1円程度を想定
（ご利用規模や用途に応じて柔軟にご相談可能です。
また、貴社システムに組み込み可能なAPIとしての提供も承っております。）

二次元創作の価値を守るという点において、日本の同人・創作文化を長年支えてこられたDLsite様にこそ、本技術を活用していただきたいと考え、真っ先にご連絡差し上げました。

お忙しいところ恐縮ではございますが、ご検討いただけましたら幸いです。
何卒よろしくお願い申し上げます。

---
石川風水
AIチェッカー開発者
https://aicheckers.net
contact@aicheckers.net
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="実際に送信する")
    args = parser.parse_args()

    print("=" * 60)
    print(f"From: {FROM_EMAIL}")
    print(f"To: {TO_EMAIL}")
    print(f"Subject: {SUBJECT}")
    print("=" * 60)
    print(BODY)
    print("=" * 60)

    if args.send:
        print("\n送信中...")
        result = resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [TO_EMAIL],
            "subject": SUBJECT,
            "text": BODY,
        })
        print(f"送信完了: {result}")
    else:
        print("\n[プレビューモード] 実際に送信するには --send オプションを付けてください")


if __name__ == "__main__":
    main()
