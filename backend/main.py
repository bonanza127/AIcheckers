"""
AIcheckers Backend API
Moonlight (DINOv3 Linear Probe) のみ使用
"""

import io
import time
import base64
import re
import hashlib
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from cachetools import LRUCache
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel


class URLAnalyzeRequest(BaseModel):
    url: str


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# グローバル変数
device = None
dinov3_model = None
dinov3_processor = None
dinov3_classifier = None

# キャッシュ: 画像ハッシュ -> 解析結果（最大10,000件、約1MB）
result_cache: LRUCache = LRUCache(maxsize=10000)

# レート制限: 1時間刻みで1枚回復、上限24枚
RATE_LIMIT_ENABLED = False  # True: 有効, False: 無効（開発中はFalse）
MAX_TOKENS = 24  # 上限24枚
RECOVERY_INTERVAL_HOURS = 1  # 1時間ごとに1枚回復

# IP -> {"tokens": int, "last_recovery": datetime}
rate_limit_data: dict[str, dict] = defaultdict(lambda: {"tokens": MAX_TOKENS, "last_recovery": datetime.now()})

# FANBOX VIP連携
import os
import json
import stripe
import secrets
from authlib.integrations.starlette_client import OAuth
from jose import jwt
from datetime import timedelta
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

FANBOX_SESSID = os.getenv("FANBOXSESSID", "")  # 環境変数から取得

# OAuth設定
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
TWITTER_CLIENT_ID = os.getenv("TWITTER_CLIENT_ID", "")
TWITTER_CLIENT_SECRET = os.getenv("TWITTER_CLIENT_SECRET", "")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

# Stripe設定
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")  # 月額300円のPrice ID
stripe.api_key = STRIPE_SECRET_KEY

# 本番/開発のURL
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://aicheckers.net")
FANBOX_CREATOR_ID = "aicheckers"  # マスターのFANBOXクリエイターID
VIP_DATA_PATH = Path("/home/techne/aicheckers/data/vip_users.json")
USERS_DATA_PATH = Path("/home/techne/aicheckers/data/users.json")

# ユーザーデータ管理
def load_users() -> dict:
    if USERS_DATA_PATH.exists():
        with open(USERS_DATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users(data: dict):
    USERS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_jwt_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except:
        return None

users_db: dict = {}  # 起動時にロード

# VIPユーザーリスト（pixiv ID -> VIP情報）
def load_vip_users() -> dict:
    if VIP_DATA_PATH.exists():
        with open(VIP_DATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_vip_users(data: dict):
    VIP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VIP_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

vip_users: dict = {}  # 起動時にロード

DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_CLASSIFIER_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
HF_TOKEN = "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"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にMoonlight (DINOv3) をロード"""
    global device, dinov3_model, dinov3_processor, dinov3_classifier, vip_users, users_db

    # ユーザーデータをロード
    users_db = load_users()
    print(f"Loaded {len(users_db)} users")

    # VIPユーザーリストをロード
    vip_users = load_vip_users()
    print(f"Loaded {len(vip_users)} VIP users")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Moonlight (DINOv3) ロード
    print(f"Loading Moonlight (DINOv3): {DINOV3_MODEL_NAME}")
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)

        dinov3_processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME, token=HF_TOKEN)
        dinov3_model = AutoModel.from_pretrained(
            DINOV3_MODEL_NAME,
            token=HF_TOKEN,
            attn_implementation="eager"
        )
        dinov3_model.to(device)
        dinov3_model.eval()

        # 分類器ロード
        if DINOV3_CLASSIFIER_PATH.exists():
            checkpoint = torch.load(DINOV3_CLASSIFIER_PATH, map_location=device)
            dinov3_classifier = nn.Linear(768, 2).to(device)
            dinov3_classifier.load_state_dict(checkpoint["classifier"])
            dinov3_classifier.eval()
            print(f"Moonlight classifier loaded! (val_acc: {checkpoint.get('val_acc', 'N/A')})")
        else:
            raise Exception(f"Classifier not found at {DINOV3_CLASSIFIER_PATH}")

        print("Moonlight loaded successfully!")
    except Exception as e:
        print(f"FATAL: Failed to load Moonlight: {e}")
        raise

    yield

    # クリーンアップ
    del dinov3_model, dinov3_processor, dinov3_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="AIcheckers API",
    description="AI-generated anime image detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定（フロントエンドからのアクセス許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://aicheckers.net",
        "https://www.aicheckers.net",
        "https://aicheckers.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# セッションミドルウェア（OAuth用）
app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET)

# OAuth クライアント設定
oauth = OAuth()

# Google OAuth
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

# Twitter/X OAuth 2.0
if TWITTER_CLIENT_ID and TWITTER_CLIENT_SECRET:
    oauth.register(
        name='twitter',
        client_id=TWITTER_CLIENT_ID,
        client_secret=TWITTER_CLIENT_SECRET,
        authorize_url='https://twitter.com/i/oauth2/authorize',
        access_token_url='https://api.twitter.com/2/oauth2/token',
        client_kwargs={'scope': 'tweet.read users.read offline.access'}
    )


def get_image_hash(image_bytes: bytes) -> str:
    """画像バイトからSHA256ハッシュを生成"""
    return hashlib.sha256(image_bytes).hexdigest()


def _recover_tokens(ip: str) -> None:
    """経過時間に基づいてトークンを回復（毎時0分刻み）"""
    data = rate_limit_data[ip]
    now = datetime.now()
    last_recovery = data["last_recovery"]

    # 最終回復時刻から現在までに経過した「時の境界」の数を計算
    # 例: 5:30に使用 → 6:00, 7:00 で2枚回復
    last_hour = last_recovery.replace(minute=0, second=0, microsecond=0)
    current_hour = now.replace(minute=0, second=0, microsecond=0)

    if current_hour > last_hour:
        hours_passed = int((current_hour - last_hour).total_seconds() // 3600)
        tokens_to_recover = hours_passed
        data["tokens"] = min(MAX_TOKENS, data["tokens"] + tokens_to_recover)
        data["last_recovery"] = current_hour


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """レート制限チェック。(許可されているか, 残り回数) を返す"""
    if not RATE_LIMIT_ENABLED:
        return True, MAX_TOKENS  # 無効時は常に許可、上限表示

    _recover_tokens(ip)
    data = rate_limit_data[ip]
    return data["tokens"] > 0, data["tokens"]


def increment_rate_limit(ip: str) -> None:
    """トークンを1消費"""
    if not RATE_LIMIT_ENABLED:
        return  # 無効時は消費しない

    _recover_tokens(ip)  # まず回復処理
    data = rate_limit_data[ip]
    if data["tokens"] > 0:
        data["tokens"] -= 1


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model": "Moonlight"
    }


@app.get("/health")
async def health_check(request: Request):
    # IPアドレス取得
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host
    _, remaining = check_rate_limit(ip)

    return {
        "status": "healthy" if dinov3_model is not None and dinov3_classifier is not None else "unhealthy",
        "model": "Moonlight",
        "device": str(device) if device else None,
        "cuda_available": torch.cuda.is_available(),
        "rate_limit": {
            "remaining": remaining,
            "limit": MAX_TOKENS
        }
    }


def generate_forensic_analysis(attention_map: np.ndarray, features: torch.Tensor, ai_prob: float, head_attentions: np.ndarray = None) -> tuple:
    """
    DINOv3の内部分析からフォレンジック風のログを生成

    Args:
        attention_map: attention map (num_patches x num_patches)
        features: CLS token特徴量 (768次元)
        ai_prob: AI確率 (0-1)
        head_attentions: ヘッド別attention [12, 196] (オプション)

    Returns:
        (logs, detected_traces): ログリストと検出痕跡サマリー
    """
    logs = []
    evidence = []  # 痕跡の根拠を収集
    is_ai = ai_prob > 0.5
    confidence = abs(ai_prob - 0.5) * 2

    entropy_ratio = None
    concentration = None
    head_diversity = None
    lr_symmetry = None
    center_ratio = None

    # 1. Attention分布の分析
    if attention_map is not None:
        flat = attention_map.flatten()
        flat_norm = flat / (flat.sum() + 1e-8)

        # エントロピー（注目の分散度）
        entropy = -np.sum(flat_norm * np.log(flat_norm + 1e-8))
        max_entropy = np.log(len(flat_norm))
        entropy_ratio = entropy / max_entropy

        # 集中度（上位10%のattentionが占める割合）
        sorted_attn = np.sort(flat_norm)[::-1]
        top_10_pct = max(1, int(len(sorted_attn) * 0.1))
        concentration = sorted_attn[:top_10_pct].sum()

        # 空間パターン分析（中央 vs エッジ）
        h, w = attention_map.shape
        h_margin, w_margin = h // 4, w // 4
        center = attention_map[h_margin:h-h_margin, w_margin:w-w_margin]
        center_ratio = center.sum() / (attention_map.sum() + 1e-8)

        # 左右対称性
        left_half = attention_map[:, :w//2].sum()
        right_half = attention_map[:, w//2:].sum()
        lr_symmetry = 1 - abs(left_half - right_half) / (left_half + right_half + 1e-8)

    # 2. ヘッド多様性分析（12ヘッドがどれだけ異なる場所を見ているか）
    if head_attentions is not None:
        head_peaks = head_attentions.argmax(axis=1)  # 各ヘッドのピーク位置
        unique_peaks = len(np.unique(head_peaks))
        head_diversity = unique_peaks / 12.0

    # 3. ログ生成（常に出力する基本情報）
    # 注目パターン（ヘッド多様性 + 中央集中度）
    pattern_parts = []
    if head_diversity is not None:
        unique_count = int(head_diversity * 12)
        pattern_parts.append(f"ヘッド多様性{head_diversity*100:.0f}%（{unique_count}/12）")
    if center_ratio is not None:
        pattern_parts.append(f"中央集中{center_ratio*100:.0f}%")
    if pattern_parts:
        logs.append(f"注目パターン: {', '.join(pattern_parts)}")

    # 集中度 + エントロピー（常に出力）
    if concentration is not None and entropy_ratio is not None:
        conc_label = "高" if concentration > 0.35 else "中" if concentration > 0.25 else "低"
        logs.append(f"注目集中度: {concentration*100:.1f}%（{conc_label}）、エントロピー{entropy_ratio:.2f}")

    # 空間バランス
    if lr_symmetry is not None:
        sym_label = "対称的" if lr_symmetry > 0.85 else "やや偏り" if lr_symmetry > 0.7 else "非対称"
        logs.append(f"空間バランス: 左右対称性{lr_symmetry*100:.0f}%（{sym_label}）")

    # 4. 判定根拠（条件付き）
    if is_ai:
        if head_diversity is not None and head_diversity < 0.5:
            evidence.append(f"ヘッド収束{head_diversity*100:.0f}%")
        if concentration is not None and concentration > 0.35:
            evidence.append(f"集中度{concentration*100:.0f}%")
        if entropy_ratio is not None and entropy_ratio < 0.80:
            evidence.append(f"低エントロピー")
    else:
        if head_diversity is not None and head_diversity > 0.6:
            evidence.append(f"ヘッド分散{head_diversity*100:.0f}%")
        if concentration is not None and concentration < 0.30:
            evidence.append(f"均等分布")
        if entropy_ratio is not None and entropy_ratio > 0.85:
            evidence.append(f"高エントロピー")

    # 5. 総合判定（常に追加）
    if confidence > 0.9:
        if is_ai:
            logs.append("総合判定: 機械学習モデル特有の生成パターンを高確度で検出")
        else:
            logs.append("総合判定: 人間の創作に特有の不規則性・個性を確認")
    elif confidence > 0.6:
        logs.append("総合判定: 判別可能な特徴を検出、中〜高確度")
    else:
        logs.append("総合判定: 境界領域のサンプル、追加検証を推奨")

    # 6. 検出された痕跡サマリー（指標に応じてバリエーション）
    if is_ai:
        trace_parts = []
        # ヘッド多様性に基づく分析
        if head_diversity is not None and head_diversity < 0.5:
            trace_parts.append(f"12ヘッド中{int(head_diversity*12)}個が同一領域に収束")
        # 集中度に基づく分析
        if concentration is not None and concentration > 0.40:
            trace_parts.append(f"Attentionの{concentration*100:.0f}%が局所領域に集中")
        elif concentration is not None and concentration > 0.30:
            trace_parts.append("中程度のAttention集中")
        # 中央集中に基づく分析
        if center_ratio is not None and center_ratio > 0.70:
            trace_parts.append("中央構図への顕著な偏重")
        # 対称性に基づく分析
        if lr_symmetry is not None and lr_symmetry > 0.90:
            trace_parts.append("高い左右対称性")

        if trace_parts:
            detected_traces = "; ".join(trace_parts)
        else:
            detected_traces = "複合的な特徴パターンから機械学習モデルによる生成と推測"
    else:
        trace_parts = []
        # ヘッド多様性に基づく分析
        if head_diversity is not None and head_diversity > 0.7:
            trace_parts.append(f"12ヘッドが{int(head_diversity*12)}箇所に分散")
        # エントロピーに基づく分析
        if entropy_ratio is not None and entropy_ratio > 0.88:
            trace_parts.append("高エントロピー（Attentionの自然な分散）")
        # 非対称性
        if lr_symmetry is not None and lr_symmetry < 0.75:
            trace_parts.append("非対称的な構図")
        # 集中度
        if concentration is not None and concentration < 0.28:
            trace_parts.append("特定領域への過度な集中なし")

        if trace_parts:
            detected_traces = "; ".join(trace_parts)
        else:
            detected_traces = "自然なAttention分布と有機的テクスチャを検出"

    return (logs if logs else ["分析完了: 明確な特徴パターンなし"], detected_traces)


async def analyze_with_dinov3(image: Image.Image) -> dict:
    """DINOv3 (ローカル) で解析 - Attention Map + フォレンジック分析付き"""
    if dinov3_model is None or dinov3_classifier is None:
        raise Exception("DINOv3 model or classifier not loaded")

    # 前処理
    inputs = dinov3_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 特徴抽出 + Attention Map
    with torch.no_grad():
        outputs = dinov3_model(**inputs, output_attentions=True)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token

        # 分類
        logits = dinov3_classifier(features)
        probs = torch.softmax(logits, dim=1)[0]
        ai_prob = probs[1].item()  # class 1 = AI

    # Attention Map生成 + フォレンジック分析
    attention_heatmap = None
    attention_map_raw = None
    head_attentions = None
    forensic_logs = []

    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        try:
            # 最後の層のattentionを取得
            last_attention = outputs.attentions[-1]  # [1, 12, 201, 201]
            attention_avg = last_attention.mean(dim=1)[0]  # [201, 201]

            # DINOv3のトークン順序: [CLS(0), REG1-4(1-4), PATCH(5-)]
            # レジスタトークン(4個)を除外してパッチのみのattentionを取得
            num_register_tokens = 4  # dinov3_model.config.num_register_tokens
            patch_start_idx = 1 + num_register_tokens  # CLS + REG を除外
            cls_attention = attention_avg[0, patch_start_idx:]  # CLSからパッチへのattention

            # ヘッド別attention（12ヘッド × 196パッチ）
            head_attentions = last_attention[0, :, 0, patch_start_idx:patch_start_idx+196].cpu().numpy()

            # グリッドサイズを計算（14x14 = 196パッチ）
            num_patches = int(np.sqrt(cls_attention.shape[0]))
            target_patches = num_patches * num_patches
            if target_patches <= cls_attention.shape[0]:
                cls_attention = cls_attention[:target_patches]
                attention_map_raw = cls_attention.reshape(num_patches, num_patches).cpu().numpy()

                # 正規化
                attention_map_norm = (attention_map_raw - attention_map_raw.min()) / (attention_map_raw.max() - attention_map_raw.min() + 1e-8)

                # 元の画像サイズにリサイズ
                original_size = image.size
                attention_resized = np.array(Image.fromarray((attention_map_norm * 255).astype(np.uint8)).resize(
                    original_size, Image.BILINEAR
                )) / 255.0

                # ヒートマップを作成
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(image)
                ax.imshow(attention_resized, cmap='jet', alpha=0.5)
                ax.axis('off')
                plt.tight_layout(pad=0)

                # Base64エンコード
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close(fig)
                buf.seek(0)
                attention_heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            import traceback
            print(f"Attention map generation failed: {e}")
            traceback.print_exc()

    # フォレンジック分析ログ生成
    forensic_logs, detected_traces = generate_forensic_analysis(attention_map_raw, features, ai_prob, head_attentions)

    ai_score = ai_prob * 100
    human_score = 100 - ai_score

    return {
        "ai_score": round(ai_score, 2),
        "human_score": round(human_score, 2),
        "confidence": round(abs(ai_prob - 0.5) * 200, 2),
        "is_ai": ai_prob > 0.5,
        "model_used": "Moonlight",
        "attention_map": attention_heatmap,
        "forensic_logs": forensic_logs,
        "detected_traces": detected_traces,
    }


@app.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="dinov3")
):
    """
    画像を解析してAI生成かどうかを判定（Moonlightのみ）
    """
    # IPアドレス取得（プロキシ対応）
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host

    # レート制限チェック
    allowed, remaining = check_rate_limit(ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "1日の上限（20枚）に達しました。明日またお試しください。"},
            headers={"X-RateLimit-Limit": str(DAILY_LIMIT), "X-RateLimit-Remaining": "0"}
        )

    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        start_time = time.time()

        # 画像読み込み
        contents = await file.read()
        image_hash = get_image_hash(contents)

        # キャッシュチェック
        if image_hash in result_cache:
            cached = result_cache[image_hash]
            # キャッシュヒット時もレート制限カウント増加
            increment_rate_limit(ip)
            _, remaining_after = check_rate_limit(ip)
            return JSONResponse(
                content={**cached, "cached": True, "filename": file.filename},
                headers={"X-RateLimit-Limit": str(DAILY_LIMIT), "X-RateLimit-Remaining": str(remaining_after)}
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Moonlight (DINOv3) で解析
        result = await analyze_with_dinov3(image)

        processing_time = time.time() - start_time

        response_data = {
            "is_ai": result["is_ai"],
            "ai_score": result["ai_score"],
            "human_score": result["human_score"],
            "confidence": result["confidence"],
            "verdict": "AI DETECTED" if result["is_ai"] else "HUMAN CONFIRMED",
            "processing_time": round(processing_time, 3),
            "filename": file.filename,
            "model_used": result["model_used"],
            "attention_map": result.get("attention_map"),
            "forensic_logs": result.get("forensic_logs", []),
            "detected_traces": result.get("detected_traces"),
        }

        # キャッシュに保存（filenameを除く）
        cache_data = {k: v for k, v in response_data.items() if k != "filename"}
        result_cache[image_hash] = cache_data

        # レート制限カウント増加
        increment_rate_limit(ip)
        _, remaining_after = check_rate_limit(ip)

        return JSONResponse(
            content=response_data,
            headers={"X-RateLimit-Limit": str(DAILY_LIMIT), "X-RateLimit-Remaining": str(remaining_after)}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def extract_twitter_image_url(url: str) -> str:
    """TwitterのURLから直接画像URLを抽出・変換"""
    # pbs.twimg.com の画像URL
    if "pbs.twimg.com" in url:
        # format=jpg&name=small などを ?format=jpg&name=orig に変換
        if "?" in url:
            base_url = url.split("?")[0]
            return f"{base_url}?format=jpg&name=orig"
        return url

    # x.com や twitter.com のツイートURLの場合
    # 例: https://x.com/user/status/123456/photo/1
    tweet_pattern = r"(?:twitter\.com|x\.com)/\w+/status/(\d+)"
    if re.search(tweet_pattern, url):
        # ツイートURLからは画像を直接取得できないのでエラー
        raise HTTPException(
            status_code=400,
            detail="ツイートURLからの画像取得はサポートされていません。画像を直接右クリックして「画像のアドレスをコピー」してください。"
        )

    return url


@app.post("/analyze-url")
async def analyze_image_from_url(request: Request, body: URLAnalyzeRequest):
    """
    URLから画像を取得して解析（Moonlightのみ）
    """
    # IPアドレス取得（プロキシ対応）
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host

    # レート制限チェック
    allowed, remaining = check_rate_limit(ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "1日の上限（20枚）に達しました。明日またお試しください。"},
            headers={"X-RateLimit-Limit": str(DAILY_LIMIT), "X-RateLimit-Remaining": "0"}
        )

    url = body.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URLが指定されていません")

    # Twitter画像URLの変換
    try:
        url = extract_twitter_image_url(url)
    except HTTPException:
        raise

    try:
        start_time = time.time()

        # 画像をダウンロード
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": url  # Pixiv対策
            }
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"画像の取得に失敗しました (HTTP {response.status_code})"
                )

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="URLが画像ではありません"
                )

            contents = response.content

        # キャッシュチェック
        image_hash = get_image_hash(contents)
        if image_hash in result_cache:
            cached = result_cache[image_hash]
            increment_rate_limit(ip)
            _, remaining_after = check_rate_limit(ip)
            filename = url.split("/")[-1].split("?")[0] or "image_from_url"
            return JSONResponse(
                content={**cached, "cached": True, "filename": filename, "source_url": body.url},
                headers={"X-RateLimit-Limit": str(DAILY_LIMIT), "X-RateLimit-Remaining": str(remaining_after)}
            )

        # 画像を開く
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Moonlight (DINOv3) で解析
        result = await analyze_with_dinov3(image)

        processing_time = time.time() - start_time

        # ファイル名をURLから抽出
        filename = url.split("/")[-1].split("?")[0] or "image_from_url"

        response_data = {
            "is_ai": result["is_ai"],
            "ai_score": result["ai_score"],
            "human_score": result["human_score"],
            "confidence": result["confidence"],
            "verdict": "AI DETECTED" if result["is_ai"] else "HUMAN CONFIRMED",
            "processing_time": round(processing_time, 3),
            "filename": filename,
            "source_url": body.url,
            "model_used": result["model_used"],
            "attention_map": result.get("attention_map"),
            "forensic_logs": result.get("forensic_logs", []),
            "detected_traces": result.get("detected_traces"),
        }

        # キャッシュに保存
        cache_data = {k: v for k, v in response_data.items() if k not in ("filename", "source_url")}
        result_cache[image_hash] = cache_data

        # レート制限カウント増加
        increment_rate_limit(ip)
        _, remaining_after = check_rate_limit(ip)

        return JSONResponse(
            content=response_data,
            headers={"X-RateLimit-Limit": str(DAILY_LIMIT), "X-RateLimit-Remaining": str(remaining_after)}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL解析失敗: {str(e)}")


# ==================== FANBOX VIP連携 ====================

class VerifyFanboxRequest(BaseModel):
    pixiv_id: str


async def fetch_fanbox_supporters() -> set[str]:
    """FANBOXダッシュボードから支援者のpixiv IDリストを取得"""
    if not FANBOX_SESSID:
        print("Warning: FANBOXSESSID not set")
        return set()

    try:
        async with httpx.AsyncClient() as client:
            # FANBOXの支援者一覧API（非公式）
            response = await client.get(
                f"https://api.fanbox.cc/relationship.listFans?status=supporter",
                headers={
                    "Accept": "application/json",
                    "Origin": "https://www.fanbox.cc",
                    "Referer": "https://www.fanbox.cc/",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                cookies={"FANBOXSESSID": FANBOX_SESSID},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                # レスポンス構造に応じてpixiv IDを抽出
                supporters = set()
                for fan in data.get("body", []):
                    pixiv_id = fan.get("user", {}).get("userId") or fan.get("userId")
                    if pixiv_id:
                        supporters.add(str(pixiv_id))
                print(f"Fetched {len(supporters)} supporters from FANBOX")
                return supporters
            else:
                print(f"FANBOX API error: {response.status_code}")
                return set()
    except Exception as e:
        print(f"FANBOX fetch error: {e}")
        return set()


@app.post("/verify-fanbox")
async def verify_fanbox(body: VerifyFanboxRequest):
    """FANBOXの支援状況を確認してVIP付与"""
    global vip_users

    pixiv_id = body.pixiv_id.strip()
    if not pixiv_id:
        raise HTTPException(status_code=400, detail="pixiv IDを入力してください")

    # 既にVIPの場合
    if pixiv_id in vip_users:
        return {"status": "already_vip", "message": "既にVIP会員です"}

    # FANBOXから支援者リストを取得
    supporters = await fetch_fanbox_supporters()

    if pixiv_id in supporters:
        # VIP付与
        vip_users[pixiv_id] = {
            "registered_at": datetime.now().isoformat(),
            "source": "fanbox"
        }
        save_vip_users(vip_users)
        return {"status": "success", "message": "VIP会員として登録されました"}
    else:
        return {"status": "not_found", "message": "FANBOXでの支援が確認できませんでした"}


@app.get("/check-vip/{pixiv_id}")
async def check_vip(pixiv_id: str):
    """VIPステータスを確認"""
    if pixiv_id in vip_users:
        return {"is_vip": True, "data": vip_users[pixiv_id]}
    return {"is_vip": False}


# ==================== Stripe決済 ====================

class CreateCheckoutRequest(BaseModel):
    email: str
    payment_method: str = "stripe"  # stripe, paypal, paypay


@app.post("/create-checkout-session")
async def create_checkout_session(body: CreateCheckoutRequest):
    """Stripe Checkout Sessionを作成"""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured")

    email = body.email.strip()
    if not email:
        raise HTTPException(status_code=400, detail="メールアドレスを入力してください")

    try:
        # PayPay/PayPalの場合もStripe経由で処理
        payment_method_types = ["card"]  # デフォルト: クレカ
        if body.payment_method == "paypay":
            payment_method_types = ["card"]  # PayPayはStripe経由でカード決済として処理
        elif body.payment_method == "paypal":
            payment_method_types = ["paypal"]  # PayPalはStripe PayPalを使用

        # Checkout Session作成
        session = stripe.checkout.Session.create(
            payment_method_types=payment_method_types,
            mode="subscription",  # 月額課金
            line_items=[{
                "price": STRIPE_PRICE_ID,
                "quantity": 1,
            }],
            customer_email=email,
            success_url=f"{FRONTEND_URL}?vip=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}?vip=cancelled",
            metadata={
                "email": email,
                "payment_method": body.payment_method,
            },
        )

        return {
            "checkout_url": session.url,
            "session_id": session.id,
        }

    except stripe.error.StripeError as e:
        print(f"Stripe error: {e}")
        raise HTTPException(status_code=500, detail=f"決済セッションの作成に失敗しました: {str(e)}")
    except Exception as e:
        print(f"Checkout session error: {e}")
        raise HTTPException(status_code=500, detail="決済セッションの作成に失敗しました")


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    """Stripe Webhookを受信"""
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # イベント処理
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_email") or session.get("metadata", {}).get("email")
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")

        if email:
            # VIP付与（メールアドレスをキーとして保存）
            global vip_users
            vip_users[email] = {
                "registered_at": datetime.now().isoformat(),
                "source": "stripe",
                "customer_id": customer_id,
                "subscription_id": subscription_id,
            }
            save_vip_users(vip_users)
            print(f"VIP granted to: {email}")

    elif event["type"] == "customer.subscription.deleted":
        # サブスク解約時
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")

        # customer_idでVIPを検索して削除
        for email, data in list(vip_users.items()):
            if data.get("customer_id") == customer_id:
                del vip_users[email]
                save_vip_users(vip_users)
                print(f"VIP revoked from: {email}")
                break

    return {"status": "ok"}


@app.get("/check-vip-email/{email}")
async def check_vip_email(email: str):
    """メールアドレスでVIPステータスを確認"""
    if email in vip_users:
        return {"is_vip": True, "data": vip_users[email]}
    return {"is_vip": False}


# ==================== OAuth認証 ====================

@app.get("/auth/google")
async def auth_google(request: Request):
    """Google OAuth開始"""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")
    redirect_uri = f"{FRONTEND_URL.replace('https://aicheckers.net', 'https://api.aicheckers.net')}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """Google OAuthコールバック"""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')

        if not user_info:
            return RedirectResponse(f"{FRONTEND_URL}?auth=error&message=userinfo_failed")

        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('sub')

        # ユーザー登録/更新
        global users_db
        user_id = f"google_{google_id}"

        if user_id not in users_db:
            users_db[user_id] = {
                "id": user_id,
                "email": email,
                "name": name,
                "provider": "google",
                "created_at": datetime.now().isoformat()
            }
            save_users(users_db)

        # JWT発行
        jwt_token = create_jwt_token(user_id, email)

        # VIPステータス確認
        is_vip = email in vip_users

        return RedirectResponse(
            f"{FRONTEND_URL}?auth=success&token={jwt_token}&name={name}&email={email}&is_vip={str(is_vip).lower()}"
        )
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return RedirectResponse(f"{FRONTEND_URL}?auth=error&message=oauth_failed")


@app.get("/auth/twitter")
async def auth_twitter(request: Request):
    """Twitter/X OAuth開始"""
    if not TWITTER_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Twitter OAuth not configured")
    redirect_uri = f"{FRONTEND_URL.replace('https://aicheckers.net', 'https://api.aicheckers.net')}/auth/twitter/callback"
    return await oauth.twitter.authorize_redirect(request, redirect_uri)


@app.get("/auth/twitter/callback")
async def auth_twitter_callback(request: Request):
    """Twitter/X OAuthコールバック"""
    try:
        token = await oauth.twitter.authorize_access_token(request)

        # Twitter APIでユーザー情報取得
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.twitter.com/2/users/me",
                headers={"Authorization": f"Bearer {token['access_token']}"},
                params={"user.fields": "id,name,username"}
            )
            if resp.status_code != 200:
                return RedirectResponse(f"{FRONTEND_URL}?auth=error&message=twitter_api_failed")

            user_data = resp.json().get("data", {})

        twitter_id = user_data.get('id')
        name = user_data.get('name', user_data.get('username'))
        username = user_data.get('username')

        # ユーザー登録/更新
        global users_db
        user_id = f"twitter_{twitter_id}"
        email = f"{username}@twitter.local"  # Twitter はメールを提供しない

        if user_id not in users_db:
            users_db[user_id] = {
                "id": user_id,
                "email": email,
                "name": name,
                "username": username,
                "provider": "twitter",
                "created_at": datetime.now().isoformat()
            }
            save_users(users_db)

        # JWT発行
        jwt_token = create_jwt_token(user_id, email)

        # VIPステータス確認（Twitterユーザーはusernameで確認）
        is_vip = email in vip_users or f"@{username}" in vip_users

        return RedirectResponse(
            f"{FRONTEND_URL}?auth=success&token={jwt_token}&name={name}&email={email}&is_vip={str(is_vip).lower()}"
        )
    except Exception as e:
        print(f"Twitter OAuth error: {e}")
        return RedirectResponse(f"{FRONTEND_URL}?auth=error&message=oauth_failed")


@app.get("/auth/me")
async def auth_me(request: Request):
    """現在のユーザー情報を取得"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header.split(" ")[1]
    payload = verify_jwt_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    email = payload.get("email")

    user = users_db.get(user_id, {})
    is_vip = email in vip_users

    return {
        "id": user_id,
        "email": email,
        "name": user.get("name", ""),
        "provider": user.get("provider", ""),
        "is_vip": is_vip
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
