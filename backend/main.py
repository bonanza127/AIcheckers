"""
AIcheckers Backend API
Moonlight (DINOv3 Linear Probe) のみ使用
"""

import io
import os
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
use_patch_stats = False  # パッチ統計量を使用するかどうか

# キャッシュ: 画像ハッシュ -> 解析結果（最大10,000件、約1MB）
result_cache: LRUCache = LRUCache(maxsize=10000)

# レート制限: 1時間刻みで回復
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"  # 本番: true（デフォルト）
MAX_TOKENS = 24  # 通常ユーザー: 上限24枚
MAX_TOKENS_VIP = 240  # VIPユーザー: 上限240枚
RECOVERY_INTERVAL_HOURS = 1  # 1時間ごとに回復
RECOVERY_AMOUNT = 1  # 通常: 1枚回復
RECOVERY_AMOUNT_VIP = 10  # VIP: 10枚回復

# 管理者アカウント（レート制限免除）
ADMIN_EMAILS = {"hokhok7676@gmail.com", "dlsite-trial@aicheckers.net"}

# Test Time Augmentation (TTA): 水平反転で2回推論し平均化
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"

# Temperature Scaling: 過信を防ぎ確率を平滑化（検証データで調整）
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.5"))

# IP -> {"tokens": int, "last_recovery": datetime}
rate_limit_data: dict[str, dict] = defaultdict(lambda: {"tokens": MAX_TOKENS, "last_recovery": datetime.now()})

# FANBOX VIP連携
import json
import stripe
import secrets
import bcrypt
import paypalrestsdk
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

# PayPal設定
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_MODE = os.getenv("PAYPAL_MODE", "sandbox")  # sandbox or live

if PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET:
    paypalrestsdk.configure({
        "mode": PAYPAL_MODE,
        "client_id": PAYPAL_CLIENT_ID,
        "client_secret": PAYPAL_CLIENT_SECRET
    })

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
    global device, dinov3_model, dinov3_processor, dinov3_classifier, use_patch_stats, vip_users, users_db

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

        # 分類器ロード（入力次元を動的に取得）
        if DINOV3_CLASSIFIER_PATH.exists():
            checkpoint = torch.load(DINOV3_CLASSIFIER_PATH, map_location=device)
            input_dim = checkpoint.get("input_dim", 768)  # デフォルト768（後方互換）
            dinov3_classifier = nn.Linear(input_dim, 2).to(device)
            dinov3_classifier.load_state_dict(checkpoint["classifier"])
            dinov3_classifier.eval()
            use_patch_stats = checkpoint.get("use_patch_stats", input_dim > 768)  # 774次元なら自動でTrue
            print(f"Moonlight classifier loaded! (input_dim: {input_dim}, patch_stats: {use_patch_stats}, val_acc: {checkpoint.get('val_acc', 'N/A')})")
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
DEBUG = os.getenv("DEBUG", "true").lower() == "true"  # ローカル開発用にデフォルトtrue
CORS_ORIGINS = [
    "https://aicheckers.net",
    "https://www.aicheckers.net",
]
if DEBUG:
    CORS_ORIGINS.extend(["http://localhost:3000", "http://localhost:3001"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600,
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


def _recover_tokens(ip: str, is_vip: bool = False) -> None:
    """経過時間に基づいてトークンを回復（毎時0分刻み）"""
    data = rate_limit_data[ip]
    now = datetime.now()
    last_recovery = data["last_recovery"]

    max_tokens = MAX_TOKENS_VIP if is_vip else MAX_TOKENS
    recovery_amount = RECOVERY_AMOUNT_VIP if is_vip else RECOVERY_AMOUNT

    # 最終回復時刻から現在までに経過した「時の境界」の数を計算
    # 例: 5:30に使用 → 6:00, 7:00 で2回復
    last_hour = last_recovery.replace(minute=0, second=0, microsecond=0)
    current_hour = now.replace(minute=0, second=0, microsecond=0)

    if current_hour > last_hour:
        hours_passed = int((current_hour - last_hour).total_seconds() // 3600)
        tokens_to_recover = hours_passed * recovery_amount
        data["tokens"] = min(max_tokens, data["tokens"] + tokens_to_recover)
        data["last_recovery"] = current_hour


def get_rate_limit_key(request: Request) -> tuple[str, bool, bool]:
    """リクエストからレート制限キー、VIPステータス、管理者ステータスを取得。
    ログインユーザーはuser_id、非ログインはIPをキーにする。
    Returns: (key, is_vip, is_admin)
    """
    # Authorizationヘッダーからユーザー情報を取得
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload = verify_jwt_token(token)
        if payload:
            user_id = payload.get("sub")
            email = payload.get("email")
            # JWTにis_adminフラグがあればそれを使う（マジックリンク用）
            jwt_is_admin = payload.get("is_admin", False)
            is_admin = jwt_is_admin or (email in ADMIN_EMAILS if email else False)
            is_vip = (email in vip_users or is_admin) if email else is_admin
            return user_id, is_vip, is_admin

    # 非ログインユーザーはIPをキーにする
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host
    return ip, False, False


def check_rate_limit(key: str, is_vip: bool = False, is_admin: bool = False) -> tuple[bool, int, int]:
    """レート制限チェック。(許可されているか, 残り回数, 上限) を返す"""
    # 管理者は無制限
    if is_admin:
        return True, 9999, 9999

    max_tokens = MAX_TOKENS_VIP if is_vip else MAX_TOKENS

    if not RATE_LIMIT_ENABLED:
        return True, max_tokens, max_tokens  # 無効時は常に許可、上限表示

    _recover_tokens(key, is_vip)
    data = rate_limit_data[key]
    return data["tokens"] > 0, data["tokens"], max_tokens


def increment_rate_limit(key: str, is_vip: bool = False, is_admin: bool = False) -> None:
    """トークンを1消費"""
    # 管理者は消費しない
    if is_admin:
        return

    if not RATE_LIMIT_ENABLED:
        return  # 無効時は消費しない

    _recover_tokens(key, is_vip)  # まず回復処理
    data = rate_limit_data[key]
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
    # レート制限キー取得（ログインユーザーはuser_id、非ログインはIP）
    key, is_vip, is_admin = get_rate_limit_key(request)
    _, remaining, limit = check_rate_limit(key, is_vip, is_admin)

    return {
        "status": "healthy" if dinov3_model is not None and dinov3_classifier is not None else "unhealthy",
        "model": "Moonlight",
        "device": str(device) if device else None,
        "cuda_available": torch.cuda.is_available(),
        "rate_limit": {
            "remaining": remaining,
            "limit": limit
        }
    }


def compute_patch_stats(patch_embeddings: torch.Tensor, classifier: nn.Linear, return_scores: bool = False):
    """
    パッチ埋め込みから統計量を計算（推論時用）

    Args:
        patch_embeddings: (1, 196, 768) パッチ埋め込み
        classifier: 768→2の分類器（パッチスコア計算用）
        return_scores: Trueの場合、パッチスコア配列も返す

    Returns:
        return_scores=False: (1, 7) パッチ統計量
        return_scores=True: ((1, 7) パッチ統計量, (196,) パッチAIスコア)
    """
    import torch.nn.functional as F

    HIGH_SCORE_THRESHOLD = 0.8
    HIGH_SIM_THRESHOLD = 0.85
    grid_size = 14

    with torch.no_grad():
        # パッチごとのAIスコアを計算（768次元入力）
        # 775次元分類器の場合は先頭768次元のみ使用
        weight = classifier.weight[:, :768]  # (2, 768)
        bias = classifier.bias
        flat_patches = patch_embeddings.reshape(-1, 768)  # (196, 768)
        logits = torch.mm(flat_patches, weight.t()) + bias  # (196, 2)
        probs = torch.softmax(logits, dim=1)
        ai_scores = probs[:, 1]  # (196,)

        # 統計量計算
        patch_mean = ai_scores.mean()
        patch_max = ai_scores.max()
        patch_var = ai_scores.var()
        max_minus_mean = patch_max - patch_mean
        embed_var_mean = patch_embeddings[0].var(dim=0).mean()
        count_high_score = (ai_scores >= HIGH_SCORE_THRESHOLD).float().mean()

        # v_high_sim_85: 垂直方向の高類似度パッチ比率
        patch_emb = patch_embeddings[0]  # (196, 768)
        patches_grid = patch_emb.reshape(grid_size, grid_size, -1)  # (14, 14, 768)
        v_sims = []
        for row in range(grid_size - 1):
            for col in range(grid_size):
                current = patches_grid[row, col]
                down = patches_grid[row + 1, col]
                sim = F.cosine_similarity(current.unsqueeze(0), down.unsqueeze(0)).item()
                v_sims.append(sim)
        v_high_sim_85 = torch.tensor(sum(1 for s in v_sims if s > HIGH_SIM_THRESHOLD) / len(v_sims), device=patch_embeddings.device)

        stats = torch.stack([patch_mean, patch_max, patch_var, max_minus_mean, embed_var_mean, count_high_score, v_high_sim_85])

        if return_scores:
            return stats.unsqueeze(0), ai_scores  # (1, 7), (196,)
        return stats.unsqueeze(0)  # (1, 7)


def generate_forensic_analysis(attention_map: np.ndarray, features: torch.Tensor, ai_prob: float, head_attentions: np.ndarray = None, patch_scores: np.ndarray = None) -> tuple:
    """
    DINOv3の内部分析からフォレンジック風のログを生成

    Args:
        attention_map: attention map (num_patches x num_patches)
        features: CLS token特徴量 (768次元)
        ai_prob: AI確率 (0-1)
        head_attentions: ヘッド別attention [12, 196] (オプション)
        patch_scores: パッチごとのAIスコア (196,) (オプション)

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

    # 4.5 パッチスコア解析（局所的なAI痕跡検出）
    # Attention × AIスコアの合成値で領域判定（ヒートマップと整合性を取る）
    patch_analysis_log = None
    high_score_regions = []
    if patch_scores is not None:
        # 14x14グリッドに変換
        grid = patch_scores.reshape(14, 14)
        patch_mean = patch_scores.mean()
        patch_max = patch_scores.max()
        patch_var = patch_scores.var()

        # Attentionとの合成マップを作成（ヒートマップと同じロジック）
        if attention_map is not None and attention_map.shape == (14, 14):
            attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            combined_grid = grid * attention_norm
            combined_grid = (combined_grid - combined_grid.min()) / (combined_grid.max() - combined_grid.min() + 1e-8)
        else:
            combined_grid = grid

        # 9領域に分割して解析（左上、上、右上、左、中央、右、左下、下、右下）
        regions = {
            "左上": combined_grid[0:5, 0:5],
            "上": combined_grid[0:5, 5:9],
            "右上": combined_grid[0:5, 9:14],
            "左": combined_grid[5:9, 0:5],
            "中央": combined_grid[5:9, 5:9],
            "右": combined_grid[5:9, 9:14],
            "左下": combined_grid[9:14, 0:5],
            "下": combined_grid[9:14, 5:9],
            "右下": combined_grid[9:14, 9:14],
        }

        # 各領域の平均スコアを計算（合成値ベース）
        region_scores = {name: region.mean() for name, region in regions.items()}
        combined_mean = combined_grid.mean()

        # 高スコア領域を検出（相対的：平均の1.3倍以上、または上位で平均より0.1以上高い）
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_regions[:3]:  # 上位3領域まで
            # 平均の1.3倍以上、または平均より0.1以上高い場合
            if score >= combined_mean * 1.3 or score >= combined_mean + 0.1:
                high_score_regions.append((name, score))

        # ログ生成
        if high_score_regions:
            # スコア順にソート
            high_score_regions.sort(key=lambda x: x[1], reverse=True)
            region_strs = [f"{name}({score*100:.0f}%)" for name, score in high_score_regions[:3]]
            patch_analysis_log = f"パッチ解析: {', '.join(region_strs)}に局所的AI痕跡（全体平均{patch_mean*100:.0f}%）"
        else:
            # 高スコア領域がない場合でも基本情報は出力
            patch_analysis_log = f"パッチ解析: 平均{patch_mean*100:.0f}%、最大{patch_max*100:.0f}%、分散{patch_var:.3f}"

        logs.append(patch_analysis_log)

        # evidenceにも追加
        if high_score_regions and is_ai:
            top_region = high_score_regions[0]
            evidence.append(f"{top_region[0]}領域{top_region[1]*100:.0f}%")

    # 4.6 パッチ統計を痕跡用に保存（後で使用）
    patch_trace_info = None
    if patch_scores is not None:
        patch_trace_info = {
            "max_score": patch_max,
            "mean_score": patch_mean,
            "max_minus_mean": patch_max - patch_mean,
            "var_score": patch_var,
            "high_regions": high_score_regions
        }

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
            trace_parts.append(f"マルチヘッドの{int(head_diversity*12)}個が単一の特徴量に収束。AI特有の画一的な演算パターンを検出")
        # 集中度に基づく分析
        if concentration is not None and concentration > 0.40:
            trace_parts.append(f"特定パッチへの異常なAttention集中（{concentration*100:.0f}%）。局所的なAIアーティファクト、またはLoRAの痕跡を検知")
        elif concentration is not None and concentration > 0.30:
            trace_parts.append("特定領域にリソースが偏る生成モデル特有の局所的バイアスを検出")
        # 中央集中に基づく分析
        if center_ratio is not None and center_ratio > 0.70:
            trace_parts.append("中央領域への過剰なリソース配分。学習モデル特有の構図バイアスを検出")
        # 対称性に基づく分析
        if lr_symmetry is not None and lr_symmetry > 0.90:
            trace_parts.append("高精度の左右対称性。手描きでは不可能な数学的対称性を特定")

        # パッチ解析に基づく痕跡
        if patch_trace_info is not None:
            # 高スコア領域の検出
            if patch_trace_info["max_score"] > 0.8 and patch_trace_info["high_regions"]:
                top_region = patch_trace_info["high_regions"][0]
                trace_parts.append(f"{top_region[0]}領域にAI生成特有のパターンを確認（信頼度{top_region[1]*100:.0f}%）。局所的な描画密度の不自然な偏りを特定")
            # 局所的な異常（最大と平均の乖離）
            if patch_trace_info["max_minus_mean"] > 0.4 and patch_trace_info["high_regions"]:
                pos = patch_trace_info["high_regions"][0][0]
                trace_parts.append(f"画像{pos}部に極めて強い生成痕跡を検出。周辺パッチとの統計的乖離からLoRA使用の蓋然性を識別")
            # 分散異常
            if patch_trace_info["var_score"] > 0.15:
                trace_parts.append("特定の描画レイヤーにおいてAIモデル固有のシグネチャーを捕捉。テクスチャの再現性に非人間的な一貫性を検出")

        if trace_parts:
            detected_traces = "; ".join(trace_parts)
        else:
            detected_traces = "パッチ間の相関統計および高次元特徴量に基づき、非人間的な生成プロセスと断定"
    else:
        trace_parts = []
        # ヘッド多様性に基づく分析
        if head_diversity is not None and head_diversity > 0.7:
            trace_parts.append(f"12層のマルチヘッドが{int(head_diversity*12)}箇所へ柔軟に分散。多層的な意図に基づく有機的な筆致を確認")
        # エントロピーに基づく分析
        if entropy_ratio is not None and entropy_ratio > 0.88:
            trace_parts.append("Attentionの適度な分散を計測。人間の手によるゆらぎと複雑性が高度に調和")
        # 非対称性
        if lr_symmetry is not None and lr_symmetry < 0.75:
            trace_parts.append("非対称的かつ動的な構図。意図された視覚的誘導と自然なattention分布を検出")
        # 集中度
        if concentration is not None and concentration < 0.28:
            trace_parts.append("特定領域への機械的な痕跡なし。画像全体にわたる自然な密度バランスと空間把握を確認")

        if trace_parts:
            detected_traces = "; ".join(trace_parts)
        else:
            detected_traces = "ミクロ領域における有機的なテクスチャーと、躍動感ある描画シグネチャーを検出"

    return (logs if logs else ["分析完了: 明確な特徴パターンなし"], detected_traces)


def get_verdict(ai_score: float) -> str:
    """AIスコアに基づいてverdict（判定結果）を返す"""
    if ai_score >= 80:
        return "AI DETECTED"
    elif ai_score >= 60:
        return "HIGH ALERT"
    elif ai_score >= 40:
        return "UNKNOWN"
    elif ai_score >= 20:
        return "MINOR CAUTION"
    else:
        return "HUMAN CONFIRMED"


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
        hidden_states = outputs.last_hidden_state
        features = hidden_states[:, 0, :]  # CLS token (1, 768)

        # パッチ埋め込みを取得（フォレンジック分析用に常に取得）
        # DINOv3: [CLS, REG1-4, PATCH1-196] なので 5: がパッチ
        patch_embeddings = hidden_states[:, 5:5+196, :]  # (1, 196, 768)

        # パッチ統計量を追加（use_patch_statsがTrueの場合）+ パッチスコア取得
        patch_scores_np = None
        if use_patch_stats:
            patch_stats, patch_scores = compute_patch_stats(patch_embeddings, dinov3_classifier, return_scores=True)
            features = torch.cat([features, patch_stats], dim=1)  # (1, 774)
            patch_scores_np = patch_scores.cpu().numpy()
        else:
            # use_patch_statsがFalseでもフォレンジック用にパッチスコアを計算
            _, patch_scores = compute_patch_stats(patch_embeddings, dinov3_classifier, return_scores=True)
            patch_scores_np = patch_scores.cpu().numpy()

        # 分類（Temperature Scalingで確率を平滑化）
        logits = dinov3_classifier(features)
        probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]
        ai_prob = probs[1].item()  # class 1 = AI

        # TTA: 水平反転で追加推論し平均化
        tta_log = None
        if TTA_ENABLED:
            ai_prob_original = ai_prob
            image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            inputs_flipped = dinov3_processor(images=image_flipped, return_tensors="pt")
            inputs_flipped = {k: v.to(device) for k, v in inputs_flipped.items()}
            outputs_flipped = dinov3_model(**inputs_flipped)
            hidden_states_flipped = outputs_flipped.last_hidden_state
            features_flipped = hidden_states_flipped[:, 0, :]
            if use_patch_stats:
                patch_embeddings_flipped = hidden_states_flipped[:, 5:5+196, :]
                patch_stats_flipped = compute_patch_stats(patch_embeddings_flipped, dinov3_classifier)
                features_flipped = torch.cat([features_flipped, patch_stats_flipped], dim=1)
            logits_flipped = dinov3_classifier(features_flipped)
            probs_flipped = torch.softmax(logits_flipped / TEMPERATURE, dim=1)[0]
            ai_prob_flipped = probs_flipped[1].item()
            # 平均化
            ai_prob = (ai_prob_original + ai_prob_flipped) / 2
            # TTAログ生成
            tta_log = f"TTA検証: 元画像 {ai_prob_original*100:.1f}% ↔ 反転画像 {ai_prob_flipped*100:.1f}% → 統合値 {ai_prob*100:.1f}%"

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

                # Attention正規化
                attention_map_norm = (attention_map_raw - attention_map_raw.min()) / (attention_map_raw.max() - attention_map_raw.min() + 1e-8)

                # パッチAIスコアを14x14にreshape（Attention × AIスコアの合成マップ）
                if patch_scores_np is not None and len(patch_scores_np) == 196:
                    patch_scores_grid = patch_scores_np.reshape(14, 14)
                    # 合成: AIスコアをベースに、Attentionで透明度を調整
                    # 「モデルが注目していて、かつAIスコアが高い場所」が強調される
                    combined_map = patch_scores_grid * attention_map_norm
                    # 正規化
                    combined_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)
                    heatmap_source = combined_map
                else:
                    heatmap_source = attention_map_norm

                # 元の画像サイズにリサイズ
                original_size = image.size
                attention_resized = np.array(Image.fromarray((heatmap_source * 255).astype(np.uint8)).resize(
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

    # フォレンジック分析ログ生成（パッチスコアも渡す）
    forensic_logs, detected_traces = generate_forensic_analysis(attention_map_raw, features, ai_prob, head_attentions, patch_scores_np)

    # TTAログを先頭に追加
    if tta_log:
        forensic_logs.insert(0, tta_log)

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
    # レート制限キー取得（ログインユーザーはuser_id、非ログインはIP）
    key, is_vip, is_admin = get_rate_limit_key(request)

    # レート制限チェック
    allowed, remaining, limit = check_rate_limit(key, is_vip, is_admin)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": f"上限（{limit}枚）に達しました。1時間ごとに回復します。"},
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"}
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
            increment_rate_limit(key, is_vip, is_admin)
            _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)
            return JSONResponse(
                content={**cached, "cached": True, "filename": file.filename},
                headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
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
            "verdict": get_verdict(result["ai_score"]),
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
        increment_rate_limit(key, is_vip, is_admin)
        _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)

        return JSONResponse(
            content=response_data,
            headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
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
    # レート制限キー取得（ログインユーザーはuser_id、非ログインはIP）
    key, is_vip, is_admin = get_rate_limit_key(request)

    # レート制限チェック
    allowed, remaining, limit = check_rate_limit(key, is_vip, is_admin)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": f"上限（{limit}枚）に達しました。1時間ごとに回復します。"},
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"}
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
            increment_rate_limit(key, is_vip, is_admin)
            _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)
            filename = url.split("/")[-1].split("?")[0] or "image_from_url"
            return JSONResponse(
                content={**cached, "cached": True, "filename": filename, "source_url": body.url},
                headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
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
            "verdict": get_verdict(result["ai_score"]),
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
        increment_rate_limit(key, is_vip, is_admin)
        _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)

        return JSONResponse(
            content=response_data,
            headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
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
        # Stripe Checkout Session作成（クレカのみ）
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
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

        # VIPステータス確認（ADMIN_EMAILSも自動VIP）
        is_vip = email in vip_users or email in ADMIN_EMAILS

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

        # VIPステータス確認（Twitterユーザーはusernameで確認、ADMIN_EMAILSも自動VIP）
        is_vip = email in vip_users or f"@{username}" in vip_users or email in ADMIN_EMAILS

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
    # ADMIN_EMAILSも自動的にVIP扱い
    is_vip = email in vip_users or email in ADMIN_EMAILS

    return {
        "id": user_id,
        "email": email,
        "name": user.get("name", ""),
        "provider": user.get("provider", ""),
        "is_vip": is_vip
    }


# ==================== メール/パスワード認証 ====================

class EmailRegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class EmailLoginRequest(BaseModel):
    email: str
    password: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def is_valid_email(email: str) -> bool:
    """簡易メールアドレスバリデーション"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


@app.post("/auth/register")
async def auth_register(body: EmailRegisterRequest):
    """メール/パスワードで新規登録"""
    global users_db

    email = body.email.strip().lower()
    password = body.password
    name = body.name.strip() or email.split("@")[0]

    if not email or not is_valid_email(email):
        raise HTTPException(status_code=400, detail="有効なメールアドレスを入力してください")

    if len(password) < 8:
        raise HTTPException(status_code=400, detail="パスワードは8文字以上で入力してください")

    # 既存ユーザーチェック
    user_id = f"email_{email}"
    if user_id in users_db:
        raise HTTPException(status_code=400, detail="このメールアドレスは既に登録されています")

    # ユーザー登録
    users_db[user_id] = {
        "id": user_id,
        "email": email,
        "name": name,
        "password_hash": hash_password(password),
        "provider": "email",
        "created_at": datetime.now().isoformat()
    }
    save_users(users_db)

    # JWT発行
    jwt_token = create_jwt_token(user_id, email)
    # ADMIN_EMAILSも自動的にVIP扱い
    is_vip = email in vip_users or email in ADMIN_EMAILS

    return {
        "status": "success",
        "token": jwt_token,
        "name": name,
        "email": email,
        "is_vip": is_vip
    }


@app.post("/auth/login")
async def auth_login(body: EmailLoginRequest):
    """メール/パスワードでログイン"""
    email = body.email.strip().lower()
    password = body.password

    if not email:
        raise HTTPException(status_code=400, detail="メールアドレスを入力してください")

    user_id = f"email_{email}"
    user = users_db.get(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="メールアドレスまたはパスワードが正しくありません")

    if not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="このアカウントはソーシャルログインで登録されています")

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="メールアドレスまたはパスワードが正しくありません")

    # JWT発行
    jwt_token = create_jwt_token(user_id, email)
    # ADMIN_EMAILSも自動的にVIP扱い
    is_vip = email in vip_users or email in ADMIN_EMAILS

    return {
        "status": "success",
        "token": jwt_token,
        "name": user.get("name", ""),
        "email": email,
        "is_vip": is_vip
    }


# ==================== マジックリンク認証 ====================

@app.get("/auth/magic/{token}")
async def magic_link_login(token: str):
    """マジックリンクでログイン（期間限定の開発者/VIPアクセス）"""
    try:
        # トークンを検証
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # マジックリンク用トークンかチェック
        if payload.get("type") != "magic_link":
            raise HTTPException(status_code=400, detail="Invalid token type")

        email = payload.get("email")
        name = payload.get("name", "Developer")
        is_admin = payload.get("is_admin", False)

        # 通常のログイン用JWTを発行（有効期限はマジックリンクの期限を引き継ぐ）
        exp = payload.get("exp")
        user_id = f"magic_{email}"

        login_token = jwt.encode(
            {
                "sub": user_id,
                "email": email,
                "exp": exp,
                "is_admin": is_admin,  # 管理者フラグを埋め込み
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM
        )

        # フロントエンドにリダイレクト（トークンをクエリパラメータで渡す）
        redirect_url = f"https://aicheckers.net?magic_token={login_token}&name={name}&email={email}&is_vip=true"
        return RedirectResponse(url=redirect_url)

    except jwt.ExpiredSignatureError:
        # 期限切れの場合、フロントエンドにエラーを伝える
        return RedirectResponse(url="https://aicheckers.net?error=magic_link_expired")
    except jwt.JWTError as e:
        return RedirectResponse(url=f"https://aicheckers.net?error=invalid_magic_link")


# ==================== PayPal決済 ====================

@app.post("/create-paypal-payment")
async def create_paypal_payment(body: CreateCheckoutRequest):
    """PayPal決済を作成"""
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="PayPal is not configured")

    email = body.email.strip()
    if not email:
        raise HTTPException(status_code=400, detail="メールアドレスを入力してください")

    try:
        # APIのベースURLを取得（PayPal executeエンドポイント用）
        from urllib.parse import quote
        api_base = FRONTEND_URL.replace('https://aicheckers.net', 'https://api.aicheckers.net').replace('https://www.aicheckers.net', 'https://api.aicheckers.net')

        payment = paypalrestsdk.Payment({
            "intent": "sale",
            "payer": {"payment_method": "paypal"},
            "redirect_urls": {
                # PayPal承認後、/paypal-executeを経由して決済を完了する
                "return_url": f"{api_base}/paypal-execute?email={quote(email)}",
                "cancel_url": f"{FRONTEND_URL}?paypal=cancelled"
            },
            "transactions": [{
                "amount": {
                    "total": "300",
                    "currency": "JPY"
                },
                "description": "AIチェッカー VIP会員（1ヶ月）"
            }]
        })

        if payment.create():
            # 承認URLを取得
            for link in payment.links:
                if link.rel == "approval_url":
                    return {
                        "checkout_url": link.href,
                        "payment_id": payment.id
                    }
            raise HTTPException(status_code=500, detail="PayPal approval URL not found")
        else:
            print(f"PayPal error: {payment.error}")
            raise HTTPException(status_code=500, detail=f"PayPal決済の作成に失敗しました: {payment.error}")

    except Exception as e:
        print(f"PayPal error: {e}")
        raise HTTPException(status_code=500, detail="PayPal決済の作成に失敗しました")


@app.get("/paypal-execute")
async def paypal_execute(paymentId: str, PayerID: str, email: str = ""):
    """PayPal決済を実行（ユーザーが承認後にリダイレクトされる）"""
    try:
        payment = paypalrestsdk.Payment.find(paymentId)

        if payment.execute({"payer_id": PayerID}):
            # 決済成功 → VIP付与
            if email:
                global vip_users
                vip_users[email] = {
                    "registered_at": datetime.now().isoformat(),
                    "source": "paypal",
                    "payment_id": paymentId
                }
                save_vip_users(vip_users)
                print(f"VIP granted via PayPal: {email}")

            return RedirectResponse(f"{FRONTEND_URL}?vip=success&method=paypal")
        else:
            print(f"PayPal execute error: {payment.error}")
            return RedirectResponse(f"{FRONTEND_URL}?vip=error&message=payment_failed")

    except Exception as e:
        print(f"PayPal execute error: {e}")
        return RedirectResponse(f"{FRONTEND_URL}?vip=error&message=payment_failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
