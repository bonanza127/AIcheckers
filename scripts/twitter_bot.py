import os
import time
import json
import requests
import tweepy
from dotenv import load_dotenv
from pathlib import Path

# .envの読み込み
load_dotenv()

# X API設定
API_KEY = os.getenv("X_API_KEY")
API_SECRET = os.getenv("X_API_SECRET")
ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("X_ACCESS_SECRET")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# バックエンド設定
BACKEND_URL = "http://localhost:8000/analyze-url"

# 履歴管理
HISTORY_FILE = Path(__file__).parent / "replied_ids.json"
ACTIVE_FLAG_FILE = Path(__file__).parent / ".bot_active"

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(replied_ids):
    with open(HISTORY_FILE, "w") as f:
        json.dump(replied_ids, f)

def get_image_url_from_tweet(client, tweet_id):
    """ツイートとその親ツイートから画像URLを探す"""
    # 自分のツイートを取得
    tweet = client.get_tweet(
        tweet_id, 
        expansions=["attachments.media_keys", "referenced_tweets.id"],
        media_fields=["url"]
    )
    
    # 1. メンションされたツイート自体に画像があるか
    if tweet.includes and "media" in tweet.includes:
        for media in tweet.includes["media"]:
            if media.type == "photo":
                return media.url
    
    # 2. 返信先（親ツイート）に画像があるか
    if tweet.data and tweet.data.referenced_tweets:
        for ref in tweet.data.referenced_tweets:
            if ref.type == "replied_to":
                parent_tweet = client.get_tweet(
                    ref.id,
                    expansions=["attachments.media_keys"],
                    media_fields=["url"]
                )
                if parent_tweet.includes and "media" in parent_tweet.includes:
                    for media in parent_tweet.includes["media"]:
                        if media.type == "photo":
                            return media.url
    return None

def main():
    if not all([API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET, BEARER_TOKEN]):
        print("ERROR: .env にX APIのキーが設定されていません。")
        return

    # APIクライアント初期化
    # API v2 (Client) と v1.1 (API - メディアアップロード等に必要)
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=API_KEY,
        consumer_secret=API_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_SECRET
    )
    
    print("AIcheckers Twitter Bot 起動...")
    replied_ids = load_history()

    while True:
        # ON/OFFフラグチェック
        if not ACTIVE_FLAG_FILE.exists():
            print("Botは現在OFFです。有効にするには 'sh scripts/bot_control.sh on' を実行してください。")
            time.sleep(60)
            continue

        try:
            # メンションを取得
            # 自分のユーザーIDを取得 (初回のみ)
            me = client.get_me()
            my_id = me.data.id
            
            mentions = client.get_users_mentions(my_id, user_auth=True)
            
            if mentions.data:
                for tweet in mentions.data:
                    if tweet.id in replied_ids:
                        continue
                    
                    print(f"新着メンション: {tweet.id} - {tweet.text}")
                    
                    # 画像URLを取得
                    image_url = get_image_url_from_tweet(client, tweet.id)
                    
                    if image_url:
                        print(f"画像検出: {image_url}")
                        
                        # バックエンドで解析
                        try:
                            res = requests.post(BACKEND_URL, json={"url": image_url})
                            if res.status_code == 200:
                                data = res.json()
                                is_ai = data["is_ai"]
                                score = data["ai_score"]
                                verdict = "AI判定" if is_ai else "人間判定"
                                
                                # リプライを送信
                                reply_text = (
                                    f"画像解析が完了しました！\n\n"
                                    f"📊 判定: {verdict}\n"
                                    f"🎯 AI確率: {score * 100:.2f}%\n\n"
                                    f"#aicheckers #AIイラストチェッカー"
                                )
                                client.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet.id)
                                print(f"返信完了: {tweet.id}")
                            else:
                                print(f"解析失敗: {res.status_code}")
                        except Exception as e:
                            print(f"バックエンド通信エラー: {e}")
                    else:
                        print("画像が見つかりませんでした。")
                        # オプション: 画像がない旨を返信することも可能
                    
                    # 処理済みとして保存
                    replied_ids.append(tweet.id)
                    save_history(replied_ids)
            
            # API制限を考慮して待機（Freeプランは制限が厳しいので1分以上推奨）
            time.sleep(90)
            
        except Exception as e:
            print(f"ループ中にエラーが発生しました: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
