"""
AIcheckers Backend API
Moonlight (DINOv3 Linear Probe) のみ使用
"""

import asyncio
import io
import os
import time
import base64
import re
import hashlib
import fcntl
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# .envファイルを読み込む (backend/.env)
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

from cachetools import LRUCache
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel

# 共通モジュール（パッチ統計計算、署名検出）
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from lib.patch_stats import compute_patch_stats_inference, compute_patch_stats_v2, compute_patch_stats_v3
from lib.cpu_stats import compute_cpu_stats
from lib.extra_stats import compute_extra_stats
from lib.boundary_stats import compute_boundary_stats

# 中間層設定（パッチ統計量v2用）
MID_LAYER_INDEX = 6  # Block 6からパッチ統計量を抽出（学習時と一致させる）

# 署名検出（オプショナル - モジュールがない場合はスキップ）
try:
    from lib.signature import detect_human_signature
    SIGNATURE_DETECTION_ENABLED = True
except ImportError:
    SIGNATURE_DETECTION_ENABLED = False
    detect_human_signature = None

# Ironclad V3.1 ポイズニング（オプショナル）
try:
    from scripts.moonknight_v3 import MoonKnightV3
    MOONKNIGHT_ENABLED = True
    moonknight_engine = None  # Lazy initialization
except ImportError as e:
    print(f"Warning: MoonKnight import failed: {e}")
    MOONKNIGHT_ENABLED = False
    moonknight_engine = None

# TrustMark透かし（オプショナル）
try:
    import trustmark
    from lib.trustmark_helper import embed_watermark, extract_watermark, create_user_watermark_mapping
    TRUSTMARK_ENABLED = True
    trustmark_encoder = None  # Lazy initialization
except ImportError as e:
    print(f"Warning: TrustMark import failed: {e}")
    TRUSTMARK_ENABLED = False
    trustmark_encoder = None


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
two_head_model = None
two_head_mode = False
two_head_gpu_v3_idx = None
two_head_cpu_v2_idx = None
two_head_cpu_v3_idx = None
two_head_use_mid_adj = False
feature_mean = None
feature_std = None
additional_scale = 1.0  # 追加特徴量のスケールファクター
gate_values = None  # 次元ごとゲート値（オプショナル）
use_patch_stats = False  # パッチ統計量を使用するかどうか
# 777d構成用インデックス
patch_indices = None
extra_indices = None
boundary_indices = None

# キャッシュ: 画像ハッシュ -> 解析結果（最大10,000件、約1MB）
result_cache: LRUCache = LRUCache(maxsize=10000)

# レート制限: 日本時間の午前0時にリセット (Midnight JST Reset)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"  # 本番: true（デフォルト）
MAX_TOKENS = 24  # 通常ユーザー: 上限24枚
MAX_TOKENS_VIP = 240  # VIPユーザー: 上限240枚
# RECOVERY_INTERVAL_HOURS = 1  # 無効化 - 午前0時リセットに変更
# RECOVERY_AMOUNT = 1  # 無効化
# RECOVERY_AMOUNT_VIP = 10  # 無効化

# Guard用レート制限 (別枠)
GUARD_MAX_TOKENS = 3
GUARD_MAX_TOKENS_VIP = 30
# GUARD_RECOVERY_INTERVAL_HOURS = 8  # 無効化 - 午前0時リセットに変更
# GUARD_RECOVERY_AMOUNT = 1  # 無効化
# GUARD_RECOVERY_AMOUNT_VIP = 5  # 無効化

# 管理者アカウント（レート制限免除）
ADMIN_EMAILS = {"hokhok7676@gmail.com", "dlsite-trial@aicheckers.net"}

# Test Time Augmentation (TTA): 水平反転＋軽い縮小で推論し平均化
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"
TTA_EXTRA_ENABLED = os.getenv("TTA_EXTRA_ENABLED", "true").lower() == "true"
TTA_EXTRA_SCALE = float(os.getenv("TTA_EXTRA_SCALE", "0.85"))

# Temperature Scaling: ECE最適化済み（2025-01-09検証: T=0.9が最良）
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.9"))

# 開発者IP（レート制限免除）
DEVELOPER_IPS = set(ip.strip() for ip in os.getenv("DEVELOPER_IPS", "").split(",") if ip.strip())

# ファイルサイズ制限（DoS攻撃対策）
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# レート制限データの永続化パス
RATE_LIMIT_DATA_PATH = PROJECT_ROOT / "data" / "rate_limits.json"

# IP -> {"tokens": int, "last_recovery": datetime}
# IP -> {"tokens": int, "last_recovery": datetime}
rate_limit_data: dict[str, dict] = defaultdict(lambda: {"tokens": MAX_TOKENS, "last_recovery": datetime.now()})
rate_limit_guard: dict[str, dict] = defaultdict(lambda: {"tokens": GUARD_MAX_TOKENS, "last_recovery": datetime.now()})

# CORSなどで使用するデバッグフラグ（ローカル開発用にデフォルトtrue）
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

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
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    # 本番環境ではJWT_SECRETが必須
    if not DEBUG:
        raise RuntimeError("CRITICAL: JWT_SECRET environment variable must be set in production!")
    # 開発環境では警告を出しつつデフォルト値を使用
    print("WARNING: JWT_SECRET not set. Using random value (tokens will be invalidated on restart)")
    JWT_SECRET = secrets.token_hex(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

# Stripe設定
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")  # 月額500円のPrice ID
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

# Discord Webhook（マジックリンク経由スキャン通知）
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
VIP_DATA_PATH = PROJECT_ROOT / "data" / "vip_users.json"
USERS_DATA_PATH = PROJECT_ROOT / "data" / "users.json"

# Enterprise API設定
ENTERPRISE_KEYS_PATH = PROJECT_ROOT / "data" / "enterprise_keys.json"
ENTERPRISE_USAGE_PATH = PROJECT_ROOT / "data" / "enterprise_usage.json"
PATROL_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "patrol_embeddings.json"

# ユーザーデータ管理
def load_users() -> dict:
    if USERS_DATA_PATH.exists():
        with open(USERS_DATA_PATH, "r") as f:
            return json.load(f)
    return {}

def save_users(data: dict):
    """ファイルロックを使った安全な保存"""
    USERS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DATA_PATH, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
        try:
            json.dump(data, f, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # ロック解放

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
    """ファイルロックを使った安全な保存"""
    VIP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VIP_DATA_PATH, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # ロック解放

vip_users: dict = {}  # 起動時にロード

# Enterprise APIキー管理
def load_enterprise_keys() -> dict:
    """Enterprise APIキーをロード"""
    if ENTERPRISE_KEYS_PATH.exists():
        with open(ENTERPRISE_KEYS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_enterprise_keys(data: dict):
    """Enterprise APIキーを保存（ファイルロック使用）"""
    ENTERPRISE_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENTERPRISE_KEYS_PATH, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
        try:
            json.dump(data, f, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # ロック解放


def load_enterprise_usage() -> dict:
    """Enterprise API使用量をロード"""
    if ENTERPRISE_USAGE_PATH.exists():
        with open(ENTERPRISE_USAGE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_enterprise_usage(data: dict):
    """Enterprise API使用量を保存（ファイルロック使用）"""
    ENTERPRISE_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENTERPRISE_USAGE_PATH, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
        try:
            json.dump(data, f, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # ロック解放


def load_rate_limits() -> tuple[dict, dict]:
    """レート制限データをロード"""
    if RATE_LIMIT_DATA_PATH.exists():
        with open(RATE_LIMIT_DATA_PATH, "r") as f:
            data = json.load(f)
            # datetimeをパース
            checker = {}
            for key, val in data.get("checker", {}).items():
                checker[key] = {
                    "tokens": val["tokens"],
                    "last_recovery": datetime.fromisoformat(val["last_recovery"])
                }
            guard = {}
            for key, val in data.get("guard", {}).items():
                guard[key] = {
                    "tokens": val["tokens"],
                    "last_recovery": datetime.fromisoformat(val["last_recovery"])
                }
            return checker, guard
    return {}, {}


def save_rate_limits(checker_data: dict, guard_data: dict):
    """レート制限データを保存（ファイルロック使用）"""
    RATE_LIMIT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # datetimeをISO形式文字列に変換
    checker_serializable = {}
    for key, val in checker_data.items():
        checker_serializable[key] = {
            "tokens": val["tokens"],
            "last_recovery": val["last_recovery"].isoformat()
        }
    guard_serializable = {}
    for key, val in guard_data.items():
        guard_serializable[key] = {
            "tokens": val["tokens"],
            "last_recovery": val["last_recovery"].isoformat()
        }

    data = {
        "checker": checker_serializable,
        "guard": guard_serializable,
        "last_saved": datetime.now().isoformat()
    }

    with open(RATE_LIMIT_DATA_PATH, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # ロック解放


def generate_api_key() -> str:
    """Enterprise APIキーを生成 (aicheckers_ent_xxxxxx)"""
    return f"aicheckers_ent_{secrets.token_hex(24)}"


def track_enterprise_usage(api_key: str, endpoint: str):
    """Enterprise API使用量を記録"""
    global enterprise_usage
    month_key = datetime.now().strftime("%Y-%m")

    if api_key not in enterprise_usage:
        enterprise_usage[api_key] = {}
    if month_key not in enterprise_usage[api_key]:
        enterprise_usage[api_key][month_key] = {"total": 0, "endpoints": {}}

    enterprise_usage[api_key][month_key]["total"] += 1
    if endpoint not in enterprise_usage[api_key][month_key]["endpoints"]:
        enterprise_usage[api_key][month_key]["endpoints"][endpoint] = 0
    enterprise_usage[api_key][month_key]["endpoints"][endpoint] += 1

    # 100リクエストごとにファイル保存（パフォーマンス考慮）
    if enterprise_usage[api_key][month_key]["total"] % 100 == 0:
        save_enterprise_usage(enterprise_usage)


enterprise_keys: dict = {}  # 起動時にロード
enterprise_usage: dict = {}  # 起動時にロード

# DINOv3モデル（ローカルディレクトリからロード）
DINOV3_MODEL_PATH = PROJECT_ROOT / "models" / "dinov3-vitb16"
DINOV3_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "dinov3_classifier.pt"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
TWO_HEAD_DIR = PROJECT_ROOT / "models" / "two_head_28d_plus_60"
TWO_HEAD_GPU_V3_IDX = [1, 3, 5, 6]  # adj_sim_var, patch_var, norm_var, norm_range
TWO_HEAD_CPU16_V2_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
TWO_HEAD_CPU20_V3_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にMoonlight (DINOv3) をロード"""
    global device, dinov3_model, dinov3_processor, dinov3_classifier, use_patch_stats, feature_mean, feature_std, vip_users, users_db
    global enterprise_keys, enterprise_usage, rate_limit_data, rate_limit_guard

    # ユーザーデータをロード
    users_db = load_users()
    print(f"Loaded {len(users_db)} users")

    # VIPユーザーリストをロード
    vip_users = load_vip_users()
    print(f"Loaded {len(vip_users)} VIP users")

    # Enterprise APIキー・使用量をロード
    enterprise_keys = load_enterprise_keys()
    enterprise_usage = load_enterprise_usage()
    print(f"Loaded {len(enterprise_keys)} enterprise API keys")

    # レート制限データをロード
    checker_limits, guard_limits = load_rate_limits()
    rate_limit_data.update(checker_limits)
    rate_limit_guard.update(guard_limits)
    print(f"Loaded rate limits: {len(rate_limit_data)} checker, {len(rate_limit_guard)} guard")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Moonlight (DINOv3) ロード（ローカルディレクトリから）
    print(f"Loading Moonlight (DINOv3) from: {DINOV3_MODEL_PATH}")

    try:
        if not DINOV3_MODEL_PATH.exists():
            raise FileNotFoundError(f"DINOv3 model not found at {DINOV3_MODEL_PATH}")
        
        dinov3_processor = AutoImageProcessor.from_pretrained(str(DINOV3_MODEL_PATH))
        dinov3_model = AutoModel.from_pretrained(
            str(DINOV3_MODEL_PATH),
            attn_implementation="eager"
        )
        dinov3_model.to(device)
        dinov3_model.eval()

        # 分類器ロード（入力次元を動的に取得）
        global MID_LAYER_INDEX, patch_indices, extra_indices, boundary_indices
        if DINOV3_CLASSIFIER_PATH.exists():
            checkpoint = torch.load(DINOV3_CLASSIFIER_PATH, map_location=device, weights_only=False)
            input_dim = checkpoint.get("input_dim", 768)  # デフォルト768（後方互換）
            dinov3_classifier = nn.Linear(input_dim, 2).to(device)
            dinov3_classifier.load_state_dict(checkpoint["classifier"])
            dinov3_classifier.eval()
            use_patch_stats = checkpoint.get("use_patch_stats", input_dim > 768)  # 774次元なら自動でTrue
            # 中間層インデックスをチェックポイントから読み込み（後方互換: デフォルト6）
            MID_LAYER_INDEX = checkpoint.get("mid_layer", MID_LAYER_INDEX)
            # 777d構成用インデックス（オプショナル）
            patch_indices = checkpoint.get("patch_indices", None)
            extra_indices = checkpoint.get("extra_indices", None)
            boundary_indices = checkpoint.get("boundary_indices", None)
            # 安全性チェック: input_dim > 775 なのにindicesが未設定は不整合
            if input_dim > 775:
                if patch_indices is None or extra_indices is None or boundary_indices is None:
                    raise Exception(f"input_dim={input_dim} requires patch_indices, extra_indices, boundary_indices in checkpoint")
            if "feature_mean" in checkpoint and "feature_std" in checkpoint:
                feature_mean = torch.tensor(checkpoint["feature_mean"], device=device, dtype=torch.float32)
                feature_std = torch.tensor(checkpoint["feature_std"], device=device, dtype=torch.float32)
            # 追加特徴量スケールファクター（オプショナル、デフォルト1.0）
            global additional_scale, gate_values
            additional_scale = checkpoint.get("additional_scale", 1.0)
            # 次元ごとゲート値（オプショナル）
            if "gate_values" in checkpoint:
                gate_values = torch.tensor(checkpoint["gate_values"], device=device, dtype=torch.float32)
            else:
                gate_values = None
            extra_info = f", extra/boundary: {extra_indices is not None}" if input_dim > 775 else ""
            scale_info = f", scale: {additional_scale}x" if additional_scale != 1.0 else ""
            gate_info = f", gates: [{', '.join(f'{g:.2f}' for g in gate_values.cpu().numpy())}]" if gate_values is not None else ""
            print(f"Moonlight classifier loaded! (input_dim: {input_dim}, patch_stats: {use_patch_stats}, mid_layer: {MID_LAYER_INDEX}{extra_info}{scale_info}{gate_info}, val_acc: {checkpoint.get('val_acc', 'N/A')})")
        else:
            print(f"[WARN] Classifier not found at {DINOV3_CLASSIFIER_PATH}")

        # Two-Head (single model) ロード
        global two_head_model, two_head_mode, two_head_use_mid_adj
        global two_head_gpu_v3_idx, two_head_cpu_v2_idx, two_head_cpu_v3_idx

        if TWO_HEAD_DIR.exists():
            try:
                model_path = TWO_HEAD_DIR / "model.pt"
                if not model_path.exists():
                    raise FileNotFoundError("two-head model.pt missing")

                config_path = TWO_HEAD_DIR / "config.json"
                cfg_gpu_dim = None
                cfg_cpu_dim = None
                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                        cfg_gpu_dim = cfg.get("gpu_dim")
                        cfg_cpu_dim = cfg.get("cpu_dim")
                    except Exception as e:
                        print(f"[WARN] Failed to read config.json: {e}")

                state_dict = torch.load(model_path, map_location=device, weights_only=False)
                sd_gpu_dim = None
                sd_cpu_dim = None
                if isinstance(state_dict, dict):
                    if "gpu_mean" in state_dict:
                        sd_gpu_dim = state_dict["gpu_mean"].shape[0]
                    if "cpu_mean" in state_dict:
                        sd_cpu_dim = state_dict["cpu_mean"].shape[0]

                gpu_dim = sd_gpu_dim or cfg_gpu_dim or 4
                cpu_dim = sd_cpu_dim or cfg_cpu_dim or 24
                if sd_gpu_dim is not None and cfg_gpu_dim is not None and sd_gpu_dim != cfg_gpu_dim:
                    print(f"[WARN] gpu_dim mismatch: config={cfg_gpu_dim}, state_dict={sd_gpu_dim} (using state_dict)")
                if sd_cpu_dim is not None and cfg_cpu_dim is not None and sd_cpu_dim != cfg_cpu_dim:
                    print(f"[WARN] cpu_dim mismatch: config={cfg_cpu_dim}, state_dict={sd_cpu_dim} (using state_dict)")

                two_head_model = TwoHeadClassifier29d(gpu_dim=gpu_dim, cpu_dim=cpu_dim).to(device)
                two_head_model.load_state_dict(state_dict)
                two_head_model.eval()

                two_head_gpu_v3_idx = TWO_HEAD_GPU_V3_IDX
                two_head_cpu_v2_idx = TWO_HEAD_CPU16_V2_IDX
                two_head_cpu_v3_idx = TWO_HEAD_CPU20_V3_IDX
                two_head_use_mid_adj = gpu_dim == 5
                two_head_mode = True
                use_patch_stats = True
                print(f"Two-head model loaded: {TWO_HEAD_DIR} (gpu_dim={gpu_dim}, cpu_dim={cpu_dim}, mid_adj={two_head_use_mid_adj})")
            except Exception as e:
                print(f"[WARN] Failed to load two-head model: {e}")
                two_head_mode = False

        print("Moonlight loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load Moonlight: {e}")
        # DINOv3ロード失敗でもサーバーは起動させる（Guard機能などは使えるように）
        dinov3_model = None
        dinov3_classifier = None

    yield

    # クリーンアップ
    # Enterprise使用量を保存
    save_enterprise_usage(enterprise_usage)
    # レート制限データを保存
    save_rate_limits(dict(rate_limit_data), dict(rate_limit_guard))
    print("Rate limits saved to disk")
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
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
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


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """学習時の統計量で特徴量を標準化（存在する場合のみ）+ 追加特徴スケーリング/ゲート"""
    if feature_mean is None or feature_std is None:
        return features
    normalized = (features - feature_mean) / feature_std
    # 追加特徴量（768:）にスケーリングまたはゲートを適用
    if normalized.shape[-1] > 768:
        normalized = normalized.clone()
        if gate_values is not None:
            # 次元ごとゲートを適用
            normalized[..., 768:] *= gate_values
        elif additional_scale != 1.0:
            # 単純スケーリングを適用
            normalized[..., 768:] *= additional_scale
    return normalized


def _letterbox_512(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """PIL画像を512レターボックス + mask生成（cpu_stats_v2用）"""
    img = image.convert("RGB")
    w, h = img.size
    scale = 512 / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if (nw, nh) != img.size:
        img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (512, 512), (128, 128, 128))
    x0 = (512 - nw) // 2
    y0 = (512 - nh) // 2
    canvas.paste(img, (x0, y0))
    mask = np.zeros((512, 512), dtype=bool)
    mask[y0:y0 + nh, x0:x0 + nw] = True
    return np.array(canvas), mask


def _compute_cpu_v2_from_image(image: Image.Image) -> np.ndarray:
    """cpu_stats_v2の18dを画像から計算"""
    from extract_cpu_stats_v2 import extract_features as _cpu_v2_extract
    img_rgb, mask = _letterbox_512(image)
    feats = _cpu_v2_extract(img_rgb, mask)
    return feats


def _compute_new3_torch(mid_patches: torch.Tensor, mid_cls: torch.Tensor) -> torch.Tensor:
    """GPU8用の新規3d (B,3)"""
    pn = F.normalize(mid_patches, dim=-1)
    cn = F.normalize(mid_cls, dim=-1)
    sim = torch.bmm(pn, pn.transpose(1, 2))
    adj = ((sim > 0.7).float() * (1 - torch.eye(sim.shape[1], device=sim.device)))
    degree = adj.sum(dim=-1)

    adj_sq = torch.bmm(adj, adj)
    triangles = (adj_sq * adj).sum(dim=(1, 2)) / 6
    possible = (degree * (degree - 1) / 2).sum(dim=-1)
    local_eff = triangles / (possible + 1e-8)

    edge_idx = list(range(14)) + list(range(14, 182, 14)) + \
               list(range(27, 196, 14)) + list(range(182, 196))
    edge_idx = list(set(edge_idx))
    interior_idx = [i for i in range(196) if i not in edge_idx]
    edge_mean = sim[:, edge_idx, :][:, :, edge_idx].mean(dim=(1, 2))
    interior_mean = sim[:, interior_idx, :][:, :, interior_idx].mean(dim=(1, 2))
    edge_gap = interior_mean - edge_mean

    cls_sims = torch.bmm(pn, cn.unsqueeze(-1)).squeeze(-1)
    cls_grid = cls_sims.view(sim.shape[0], 14, 14)
    coords = torch.stack(torch.meshgrid(
        torch.arange(14, device=sim.device, dtype=torch.float32),
        torch.arange(14, device=sim.device, dtype=torch.float32),
        indexing="ij"
    ), dim=-1)
    center = torch.tensor([6.5, 6.5], device=sim.device)
    dist_from_center = ((coords - center) ** 2).sum(dim=-1).sqrt().flatten()
    center_corr = []
    for b in range(cls_grid.shape[0]):
        cls_flat = cls_grid[b].flatten()
        r = torch.corrcoef(torch.stack([dist_from_center, cls_flat]))[0, 1]
        center_corr.append(0.0 if torch.isnan(r) else r)
    center_corr = torch.stack(center_corr)

    return torch.stack([local_eff, edge_gap, center_corr], dim=1)


def compute_mid_adj_sim_var(patches: torch.Tensor) -> torch.Tensor:
    """中間層パッチから隣接類似度分散 (B,1) を計算"""
    bsz, _, dim = patches.shape
    grid = patches.reshape(bsz, 14, 14, dim)
    h_sim = F.cosine_similarity(
        grid[:, :, :-1].reshape(-1, dim),
        grid[:, :, 1:].reshape(-1, dim),
        dim=1
    ).reshape(bsz, 14, 13)
    v_sim = F.cosine_similarity(
        grid[:, :-1, :].reshape(-1, dim),
        grid[:, 1:, :].reshape(-1, dim),
        dim=1
    ).reshape(bsz, 13, 14)
    all_sim = torch.cat([h_sim.reshape(bsz, -1), v_sim.reshape(bsz, -1)], dim=1)
    return all_sim.var(dim=1, keepdim=True)


class TwoHeadClassifier29d(nn.Module):
    """Two-Head classifier: CLS 768d + GPU {gpu_dim}d + CPU {cpu_dim}d"""

    def __init__(self, cls_dim=768, gpu_dim=4, cpu_dim=24, hidden_dim=256):
        super().__init__()
        total_dim = cls_dim + gpu_dim + cpu_dim
        self.bn_input = nn.BatchNorm1d(total_dim)
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.register_buffer("cls_mean", torch.zeros(cls_dim))
        self.register_buffer("cls_std", torch.ones(cls_dim))
        self.register_buffer("gpu_mean", torch.zeros(gpu_dim))
        self.register_buffer("gpu_std", torch.ones(gpu_dim))
        self.register_buffer("cpu_mean", torch.zeros(cpu_dim))
        self.register_buffer("cpu_std", torch.ones(cpu_dim))

    def forward(self, cls_feat, gpu_feat, cpu_feat):
        std_floor = 1e-3
        cls_norm = (cls_feat - self.cls_mean) / (torch.clamp(self.cls_std, min=std_floor) + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (torch.clamp(self.gpu_std, min=std_floor) + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (torch.clamp(self.cpu_std, min=std_floor) + 1e-8)
        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


async def validate_file_size(file: UploadFile) -> bytes:
    """ファイルサイズを検証し、大きすぎる場合は例外を投げる"""
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"ファイルサイズが大きすぎます（上限: {MAX_FILE_SIZE // (1024 * 1024)}MB）"
        )
    return contents


def _recover_tokens(ip: str, is_vip: bool = False, limit_type: str = "checker") -> None:
    """日本時間の午前0時にトークンをリセット (Midnight JST Reset)"""
    import pytz
    
    if limit_type == "guard":
        data = rate_limit_guard[ip]
        max_tokens = GUARD_MAX_TOKENS_VIP if is_vip else GUARD_MAX_TOKENS
    else:
        # Default Checker
        data = rate_limit_data[ip]
        max_tokens = MAX_TOKENS_VIP if is_vip else MAX_TOKENS

    # 日本時間で現在時刻と最終リセット日を取得
    jst = pytz.timezone('Asia/Tokyo')
    now_jst = datetime.now(jst)
    
    # last_recoveryがtimezone-naiveの場合はJSTとして扱う
    last_recovery = data["last_recovery"]
    if last_recovery.tzinfo is None:
        last_recovery = jst.localize(last_recovery)
    else:
        last_recovery = last_recovery.astimezone(jst)
    
    # 日付が変わったかチェック（午前0時を跨いだか）
    today_midnight_jst = now_jst.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if last_recovery < today_midnight_jst:
        # 日付が変わった → フルリセット
        data["tokens"] = max_tokens
        data["last_recovery"] = now_jst.replace(tzinfo=None)  # Store as naive for compatibility


def verify_enterprise_api_key(request: Request) -> dict | None:
    """Enterprise APIキーを検証。有効なら企業情報を返す、無効ならNone。"""
    api_key = request.headers.get("X-API-Key", "")
    if not api_key or not api_key.startswith("aicheckers_ent_"):
        return None

    key_info = enterprise_keys.get(api_key)
    if not key_info:
        return None

    # 有効期限チェック（設定されている場合）
    expires_at = key_info.get("expires_at")
    if expires_at:
        if datetime.fromisoformat(expires_at) < datetime.now():
            return None

    # 無効化されていないかチェック
    if key_info.get("revoked", False):
        return None

    return {**key_info, "api_key": api_key}


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
            
            # マジックリンクユーザー(magic_*) または 管理者メール または JWT管理者フラグがあれば管理者(無制限)
            is_magic_user = user_id.startswith("magic_")
            email_match = (email.lower() in {e.lower() for e in ADMIN_EMAILS}) if email else False
            
            # HARDCODED CHECK for safety
            if email and "hokhok7676" in email.lower():
                email_match = True
            
            is_admin = jwt_is_admin or is_magic_user or email_match
            
            # DEBUG LOG
            if email or is_magic_user:
                print(f"[AUTH DEBUG] User: {email} (ID: {user_id}), IsAdmin: {is_admin} (JWT:{jwt_is_admin}, Magic:{is_magic_user}, Email:{email_match})")
                
            is_vip = (email in vip_users or is_admin) if email else is_admin
            return user_id, is_vip, is_admin

    # 非ログインユーザーはIPをキーにする
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.client.host
    
    # ローカル開発環境(localhost)からのアクセスは管理者扱いとする
    if ip in ["127.0.0.1", "::1", "localhost"]:
        return ip, True, True
    
    # 開発者IP（環境変数 DEVELOPER_IPS で設定）は管理者扱い
    if ip in DEVELOPER_IPS:
        print(f"[AUTH DEBUG] Developer IP detected: {ip}")
        return ip, True, True

    return ip, False, False


def get_magic_link_email(request: Request) -> str | None:
    """マジックリンク経由のリクエストならメールアドレスを返す、それ以外はNone"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload = verify_jwt_token(token)
        if payload:
            user_id = payload.get("sub", "")
            if user_id.startswith("magic_"):
                return payload.get("email")
    return None


def check_rate_limit(key: str, is_vip: bool = False, is_admin: bool = False, limit_type: str = "checker") -> tuple[bool, int, int]:
    """レート制限チェック。(許可されているか, 残り回数, 上限) を返す"""
    # 管理者は無制限（-1で無限を示す）
    if is_admin:
        return True, -1, -1

    if limit_type == "guard":
        max_tokens = GUARD_MAX_TOKENS_VIP if is_vip else GUARD_MAX_TOKENS
        data_store = rate_limit_guard
    else:
        max_tokens = MAX_TOKENS_VIP if is_vip else MAX_TOKENS
        data_store = rate_limit_data

    if not RATE_LIMIT_ENABLED:
        return True, max_tokens, max_tokens  # 無効時は常に許可、上限表示

    _recover_tokens(key, is_vip, limit_type=limit_type)
    data = data_store[key]
    return data["tokens"] > 0, data["tokens"], max_tokens


def increment_rate_limit(key: str, is_vip: bool = False, is_admin: bool = False, limit_type: str = "checker") -> None:
    """トークンを1消費"""
    # 管理者は消費しない
    if is_admin:
        return

    if not RATE_LIMIT_ENABLED:
        return  # 無効時は消費しない

    if limit_type == "guard":
        data_store = rate_limit_guard
    else:
        data_store = rate_limit_data

    _recover_tokens(key, is_vip, limit_type=limit_type)  # まず回復処理
    data = data_store[key]
    if data["tokens"] > 0:
        data["tokens"] -= 1


async def send_discord_notification(email: str, ai_score: float, verdict: str, filename: str):
    """マジックリンク経由のスキャン時にDiscord通知を送信"""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient() as client:
            embed = {
                "title": "Magic Link Scan",
                "color": 0xFF6B6B if ai_score >= 50 else 0x4ECDC4,
                "fields": [
                    {"name": "User", "value": email, "inline": True},
                    {"name": "Score", "value": f"{ai_score:.1f}%", "inline": True},
                    {"name": "Verdict", "value": verdict, "inline": True},
                    {"name": "File", "value": filename or "N/A", "inline": False},
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }
            await client.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=5.0)
    except Exception as e:
        print(f"Discord notification failed: {e}")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model": "Moonlight"
    }


@app.get("/robots.txt")
async def robots_txt():
    """検索エンジンのクロールを拒否"""
    return PlainTextResponse("User-agent: *\nDisallow: /")


@app.get("/health")
def health_check(request: Request):
    """ヘルスチェック + レート制限状態取得"""
    # アクティブなモデルを返す
    model_name = "Moonlight (DINOv3)"
    if MOONKNIGHT_ENABLED:
        model_name += " & MoonKnight V3"
    
    # レート制限情報 (Checker)
    key, is_vip, is_admin = get_rate_limit_key(request)
    _, remaining, limit = check_rate_limit(key, is_vip, is_admin, limit_type="checker")
    
    # レート制限情報 (Guard)
    _, guard_remaining, guard_limit = check_rate_limit(key, is_vip, is_admin, limit_type="guard")

    return {
        "status": "online",
        "model": model_name,
        "rate_limit": {
            "limit": limit,
            "remaining": remaining,
            "guard_limit": guard_limit,
            "guard_remaining": guard_remaining,
            "is_vip": is_vip,
            "is_admin": is_admin
        }
    }


def compute_patch_stats(patch_embeddings: torch.Tensor, classifier: nn.Linear, return_scores: bool = False):
    """パッチ統計量計算（共通モジュールへの委譲）"""
    return compute_patch_stats_inference(patch_embeddings, classifier, return_scores)


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
        patch_scores = np.asarray(patch_scores).reshape(-1)
        if patch_scores.size != 196:
            patch_scores = None

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
            trace_parts.append("生成モデル特有の局所的バイアスを検出")
        # 中央集中に基づく分析
        if center_ratio is not None and center_ratio > 0.70:
            trace_parts.append("中央領域に構図バイアスを検出")
        # 対称性に基づく分析
        if lr_symmetry is not None and lr_symmetry > 0.90:
            trace_parts.append("手描きでは不可能な数学的対称性を特定")

        # パッチ解析に基づく痕跡
        if patch_trace_info is not None:
            # 高スコア領域の検出（max-mean条件と統合）
            if patch_trace_info["max_score"] > 0.8 and patch_trace_info["high_regions"]:
                top_region = patch_trace_info["high_regions"][0]
                if patch_trace_info["max_minus_mean"] > 0.4:
                    # 局所的な異常が強い場合
                    trace_parts.append(f"{top_region[0]}領域に強い生成痕跡を検出")
                else:
                    trace_parts.append(f"{top_region[0]}領域にAI生成特有のパターンを確認")
            # 分散異常
            if patch_trace_info["var_score"] > 0.15:
                trace_parts.append("テクスチャに非人間的な一貫性を検出")

        if trace_parts:
            detected_traces = "。".join(trace_parts)
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
            detected_traces = "。".join(trace_parts)
        else:
            detected_traces = "ミクロ領域における有機的なテクスチャーと、躍動感ある描画シグネチャーを検出"

    return (logs if logs else ["分析完了: 明確な特徴パターンなし"], detected_traces)


def get_verdict(ai_score: float) -> str:
    """AIスコアに基づいてverdict（判定結果）を返す

    5段階分類（低い順）:
    - HUMAN CONFIRMED (0-20%): 青
    - LOW SIMILARITY (20-40%): 緑
    - MIDDLE CAUTION (40-60%): 黄色
    - HIGH ALERT (60-80%): オレンジ
    - AI DETECTED (80-100%): 赤
    """
    if ai_score >= 80:
        return "AI DETECTED"
    elif ai_score >= 60:
        return "HIGH ALERT"
    elif ai_score >= 40:
        return "MIDDLE CAUTION"
    elif ai_score >= 20:
        return "LOW SIMILARITY"
    else:
        return "HUMAN CONFIRMED"


def extract_dinov3_embedding(image: Image.Image) -> np.ndarray:
    """
    DINOv3のCLSトークン（768次元）を抽出

    Args:
        image: RGB画像

    Returns:
        768次元のnumpy配列
    """
    if dinov3_model is None:
        raise Exception("DINOv3 model not loaded")

    inputs = dinov3_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = dinov3_model(**inputs)
        hidden_states = outputs.last_hidden_state
        cls_token = hidden_states[:, 0, :]  # CLS token (1, 768)

    return cls_token.cpu().numpy()[0]  # (768,)


def save_patrol_embedding(user_id: str, embedding: np.ndarray, watermark_hash: str):
    """
    Patrol用にDINOv3埋め込み + タイムスタンプ + 透かしハッシュをDB保存

    Args:
        user_id: ユーザーID（Enterprise APIキー or レート制限キー）
        embedding: 768次元のDINOv3埋め込み
        watermark_hash: 61bitのバイナリ文字列（TrustMark透かし）
    """
    db_path = PATROL_EMBEDDINGS_PATH

    # データ構造: {user_id: [{embedding: [...], timestamp: "...", watermark_hash: "..."}]}
    data = {}

    # ファイルロック付き読み込み
    if db_path.exists():
        with open(db_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # 新しいエントリを追加
    if user_id not in data:
        data[user_id] = []

    data[user_id].append({
        "embedding": embedding.tolist(),  # numpy -> list
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "watermark_hash": watermark_hash
    })

    # ファイルロック付き書き込み
    with open(db_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


async def analyze_with_dinov3(image: Image.Image) -> dict:
    """DINOv3 (ローカル) で解析 - Attention Map + フォレンジック分析付き
    
    v2.1: 中間層から教師なしパッチ統計量を抽出
    """
    if dinov3_model is None:
        raise Exception("DINOv3 model not loaded")
    if not two_head_mode and dinov3_classifier is None:
        raise Exception("DINOv3 classifier not loaded")

    # 前処理
    inputs = dinov3_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 特徴抽出 + Attention Map + 中間層
    with torch.inference_mode():
        outputs = dinov3_model(**inputs, output_attentions=True, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        features = hidden_states[:, 0, :]  # CLS token (1, 768)

        # 最終層のパッチ埋め込み（ヒートマップ可視化用）
        # DINOv3: [CLS, REG1-4, PATCH1-196] なので 5: がパッチ
        patch_embeddings_final = hidden_states[:, 5:5+196, :]  # (1, 196, 768)

        # 中間層のパッチ埋め込み（統計量v2用）
        mid_hidden = outputs.hidden_states[MID_LAYER_INDEX + 1]  # +1 because index 0 is initial embedding
        patch_embeddings_mid = mid_hidden[:, 5:5+196, :]  # (1, 196, 768)
        mid_cls = mid_hidden[:, 0, :]  # (1, 768)

        # パッチ統計量v2（フォレンジック用ヒートマップは常に計算）
        patch_stats_v2, heatmap_v2 = compute_patch_stats_v2(patch_embeddings_mid, return_heatmap=True)
        patch_scores_np = heatmap_v2.detach().flatten().cpu().numpy()  # 可視化・ログ用

        # two-head以外: patch_stats_v2を特徴量に結合
        if use_patch_stats and not two_head_mode:

            # 777d構成: patch[indices] + extra[indices] + boundary[indices]
            if patch_indices is not None and extra_indices is not None and boundary_indices is not None:
                # パッチ統計から選択
                patch_stats_np = patch_stats_v2.cpu().numpy()[0]  # (7,)
                patch_sel = patch_stats_np[patch_indices]  # (2,)

                # 元画像からextra_stats, boundary_statsを計算
                img_rgb = np.array(image.convert("RGB"))
                extra_stats_full = compute_extra_stats(img_rgb)  # (15,)
                boundary_stats_full = compute_boundary_stats(img_rgb)  # (5,)

                extra_sel = extra_stats_full[extra_indices]  # (5,)
                boundary_sel = boundary_stats_full[boundary_indices]  # (2,)

                # 追加特徴量をテンソル化して結合
                additional_stats = np.concatenate([patch_sel, extra_sel, boundary_sel])  # (9,)
                additional_tensor = torch.tensor(additional_stats, dtype=torch.float32, device=device).unsqueeze(0)
                features = torch.cat([features, additional_tensor], dim=1)  # (1, 777)
            else:
                # 旧775d構成: そのまま全パッチ統計を結合
                features = torch.cat([features, patch_stats_v2], dim=1)  # (1, 775)

        # 分類
        if two_head_mode:
            cpu_v2_18d, cpu_v3_20d = compute_cpu_stats(image)
            cpu_13d = cpu_v2_18d[two_head_cpu_v2_idx]
            cpu_11d = cpu_v3_20d[two_head_cpu_v3_idx]
            cpu_24d = np.concatenate([cpu_13d, cpu_11d]).astype(np.float32)
            cpu_tensor = torch.tensor(cpu_24d, dtype=torch.float32, device=device).unsqueeze(0)

            stats_v3 = compute_patch_stats_v3(patch_embeddings_mid, mid_cls)
            gpu_4d = stats_v3[:, two_head_gpu_v3_idx]
            if two_head_use_mid_adj:
                mid_adj_var = compute_mid_adj_sim_var(patch_embeddings_mid)
                gpu_features = torch.cat([gpu_4d, mid_adj_var], dim=1)
            else:
                gpu_features = gpu_4d

            cls_features = hidden_states[:, 0, :]
            logits = two_head_model(cls_features, gpu_features, cpu_tensor)
            ai_prob = torch.sigmoid(logits)[0].item()
        else:
            features = normalize_features(features)
            logits = dinov3_classifier(features)
            if DEBUG:
                print(f"[DEBUG] CLS mean: {features[0, :768].mean().item():.4f}, var: {features[0, :768].var().item():.4f}")
                if use_patch_stats:
                    print(f"[DEBUG] Stats: {features[0, 768:].tolist()}")
                print(f"[DEBUG] Logits: {logits.tolist()}")

            probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]
            ai_prob = probs[1].item()  # class 1 = AI

        # TTA: 水平反転＋軽い縮小で追加推論し平均化
        tta_log = None
        if TTA_ENABLED:
            def infer_ai_prob(image_tta: Image.Image) -> float:
                inputs_tta = dinov3_processor(images=image_tta, return_tensors="pt")
                inputs_tta = {k: v.to(device) for k, v in inputs_tta.items()}
                outputs_tta = dinov3_model(**inputs_tta, output_hidden_states=True)
                hidden_tta = outputs_tta.last_hidden_state
                features_tta = hidden_tta[:, 0, :]
                mid_hidden_tta = outputs_tta.hidden_states[MID_LAYER_INDEX + 1]
                patch_embeddings_mid_tta = mid_hidden_tta[:, 5:5+196, :]
                mid_cls_tta = mid_hidden_tta[:, 0, :]

                if two_head_mode:
                    cpu_v2_tta, cpu_v3_tta = compute_cpu_stats(image_tta)
                    cpu_13d_tta = cpu_v2_tta[two_head_cpu_v2_idx]
                    cpu_11d_tta = cpu_v3_tta[two_head_cpu_v3_idx]
                    cpu_24d_tta = np.concatenate([cpu_13d_tta, cpu_11d_tta]).astype(np.float32)
                    cpu_tensor_tta = torch.tensor(cpu_24d_tta, dtype=torch.float32, device=device).unsqueeze(0)

                    stats_v3_tta = compute_patch_stats_v3(patch_embeddings_mid_tta, mid_cls_tta)
                    gpu_4d_tta = stats_v3_tta[:, two_head_gpu_v3_idx]
                    if two_head_use_mid_adj:
                        mid_adj_var_tta = compute_mid_adj_sim_var(patch_embeddings_mid_tta)
                        gpu_features_tta = torch.cat([gpu_4d_tta, mid_adj_var_tta], dim=1)
                    else:
                        gpu_features_tta = gpu_4d_tta

                    logits_tta = two_head_model(features_tta, gpu_features_tta, cpu_tensor_tta)
                    return torch.sigmoid(logits_tta)[0].item()

                # fallback: legacy classifier
                if use_patch_stats:
                    patch_stats_v2_tta = compute_patch_stats_v2(patch_embeddings_mid_tta)
                    if patch_indices is not None and extra_indices is not None and boundary_indices is not None:
                        patch_stats_np_tta = patch_stats_v2_tta.cpu().numpy()[0]
                        patch_sel_tta = patch_stats_np_tta[patch_indices]
                        img_rgb_tta = np.array(image_tta.convert("RGB"))
                        extra_sel_tta = compute_extra_stats(img_rgb_tta)[extra_indices]
                        boundary_sel_tta = compute_boundary_stats(img_rgb_tta)[boundary_indices]
                        additional_tta = np.concatenate([patch_sel_tta, extra_sel_tta, boundary_sel_tta])
                        additional_tensor_tta = torch.tensor(additional_tta, dtype=torch.float32, device=device).unsqueeze(0)
                        features_tta = torch.cat([features_tta, additional_tensor_tta], dim=1)
                    else:
                        features_tta = torch.cat([features_tta, patch_stats_v2_tta], dim=1)
                features_tta = normalize_features(features_tta)
                logits_tta = dinov3_classifier(features_tta)
                probs_tta = torch.softmax(logits_tta / TEMPERATURE, dim=1)[0]
                return probs_tta[1].item()

            ai_prob_original = ai_prob
            tta_probs = [("元画像", ai_prob_original)]

            image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            ai_prob_flipped = infer_ai_prob(image_flipped)
            tta_probs.append(("反転画像", ai_prob_flipped))

            if TTA_EXTRA_ENABLED:
                width, height = image.size
                scale = max(0.5, min(1.0, TTA_EXTRA_SCALE))
                scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                image_scaled = image.resize(scaled_size, Image.LANCZOS).resize((width, height), Image.LANCZOS)
                ai_prob_scaled = infer_ai_prob(image_scaled)
                tta_probs.append((f"縮小{scale:.2f}", ai_prob_scaled))

            ai_prob = sum(prob for _, prob in tta_probs) / len(tta_probs)
            tta_log_parts = [f"{label} {prob*100:.1f}%" for label, prob in tta_probs]
            tta_log = f"TTA検証: {' / '.join(tta_log_parts)} → 統合値 {ai_prob*100:.1f}%"

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
            num_register_tokens = getattr(dinov3_model.config, "num_register_tokens", 4)
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
                if patch_scores_np is not None and patch_scores_np.size == 196:
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

    # Human Signature 検出（オプショナル）
    human_verified = False
    signature_correlation = None
    if SIGNATURE_DETECTION_ENABLED and detect_human_signature:
        try:
            sig_result = detect_human_signature(image, normalize_resolution=True)
            human_verified = sig_result["detected"]
            signature_correlation = sig_result["correlation"]
            if human_verified:
                forensic_logs.append(f"✓ Human Verified署名を検出 (相関: {signature_correlation:.3f})")
                detected_traces.append("human_signature")
        except Exception as e:
            print(f"Signature detection failed: {e}")

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
        "human_verified": human_verified,
        "signature_correlation": round(signature_correlation, 4) if signature_correlation else None,
    }


@app.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="dinov3")
):
    """
    画像を解析してAI生成かどうかを判定（Moonlightのみ）

    Enterprise API: X-API-Keyヘッダーで認証（レート制限なし）
    """
    # Enterprise APIキー認証チェック（優先）
    enterprise_info = verify_enterprise_api_key(request)
    is_enterprise = enterprise_info is not None

    if is_enterprise:
        # Enterprise APIはレート制限なし
        remaining, limit = -1, -1
        track_enterprise_usage(enterprise_info["api_key"], "/analyze")
    else:
        # 通常ユーザーのレート制限
        key, is_vip, is_admin = get_rate_limit_key(request)
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

        # 画像読み込み（サイズチェック含む）
        contents = await validate_file_size(file)
        image_hash = get_image_hash(contents)

        # キャッシュチェック
        if image_hash in result_cache:
            cached = result_cache[image_hash]
            # キャッシュヒット時もレート制限カウント増加（Enterprise以外）
            if not is_enterprise:
                increment_rate_limit(key, is_vip, is_admin)
                _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)
            else:
                remaining_after, limit_after = -1, -1
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

        # レート制限カウント増加（Enterprise以外）
        if not is_enterprise:
            increment_rate_limit(key, is_vip, is_admin)
            _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)
        else:
            remaining_after, limit_after = -1, -1

        # マジックリンク経由ならDiscord通知（非同期でバックグラウンド実行）
        magic_email = get_magic_link_email(request)
        if magic_email:
            asyncio.create_task(send_discord_notification(
                magic_email,
                response_data["ai_score"],
                response_data["verdict"],
                file.filename or ""
            ))

        return JSONResponse(
            content=response_data,
            headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        if DEBUG:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="画像の解析中にエラーが発生しました。しばらく経ってから再度お試しください。")


@app.post("/guard")
async def guard_image(
    request: Request,
    file: UploadFile = File(...),
    iterations: int = Form(default=30),
    strength: float = Form(default=0.6)
):
    """
    画像にMoonKnight V3（旧FastProtect）を適用して保護する

    Returns:
        - protected_image: Base64エンコードされた保護済みPNG画像
        - processing_time: 処理時間（秒）
        - ssim: 元画像との構造類似性（品質指標）
    """
    global moonknight_engine

    if not MOONKNIGHT_ENABLED:
        raise HTTPException(status_code=503, detail="MoonKnight module not available")

    # Enterprise APIキー認証チェック（優先）
    enterprise_info = verify_enterprise_api_key(request)
    is_enterprise = enterprise_info is not None

    if is_enterprise:
        remaining, limit = -1, -1
        track_enterprise_usage(enterprise_info["api_key"], "/guard")
    else:
        key, is_vip, is_admin = get_rate_limit_key(request)
        allowed, remaining, limit = check_rate_limit(key, is_vip, is_admin, limit_type="guard")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": f"上限（{limit}枚）に達しました。1時間ごとに回復します。"},
                headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"}
            )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        start_time = time.time()

        # 画像読み込み（サイズチェック含む）
        contents = await validate_file_size(file)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_image = image.copy()

        # ユーザーID取得（Enterprise APIキー or レート制限キー）
        if is_enterprise:
            user_id = enterprise_info["api_key"]
        else:
            user_id = key

        # TrustMark透かし埋め込み（alpha=1.15、FastProtect前）
        watermark_hash = None
        if TRUSTMARK_ENABLED:
            global trustmark_encoder
            if trustmark_encoder is None:
                import trustmark as tm
                trustmark_encoder = tm.TrustMark()

            # タイムスタンプ生成
            timestamp_str = datetime.utcnow().isoformat() + "Z"

            # 透かし埋め込み
            image = embed_watermark(
                trustmark_encoder,
                image,
                user_id,
                timestamp_str,
                alpha=1.15
            )

            # 透かしハッシュ生成（DB保存用）
            watermark_hash = create_user_watermark_mapping(
                user_id,
                timestamp_str,
                capacity=trustmark_encoder.schemaCapacity()
            )

        # MoonKnightエンジンの遅延初期化
        if moonknight_engine is None:
            moonknight_engine = MoonKnightV3(
                model_dir="/home/techne/aicheckers/models/fastprotect",
                device=str(device),
                use_adaptive=True
            )

        # 保護実行（PIL -> PIL）
        # MoonKnightはiterationsパラメータを使用しません（ワンショット生成）
        protected_image = moonknight_engine.poison(image, strength=strength)

        # 変数名の互換性のため（以降の処理で使用）
        poisoned_image = protected_image

        # DINOv3埋め込み抽出（最終画像から）
        if TRUSTMARK_ENABLED and watermark_hash is not None:
            embedding = extract_dinov3_embedding(poisoned_image)
            save_patrol_embedding(user_id, embedding, watermark_hash)

        # SSIM計算（品質検証）
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        orig_arr = np.array(original_image)
        pois_arr = np.array(poisoned_image.resize(original_image.size))
        ssim_value = ssim(orig_arr, pois_arr, channel_axis=2, data_range=255)

        # Base64エンコード
        buffer = io.BytesIO()
        poisoned_image.save(buffer, format="PNG", quality=95)
        protected_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        processing_time = time.time() - start_time

        # レート制限カウント増加
        if not is_enterprise:
            increment_rate_limit(key, is_vip, is_admin, limit_type="guard")
            _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin, limit_type="guard")
        else:
            remaining_after, limit_after = -1, -1

        return JSONResponse(
            content={
                "protected_image": protected_base64,
                "processing_time": round(processing_time, 3),
                "ssim": round(ssim_value, 4),
                "filename": file.filename,
                "protection_applied": "MoonKnight V3",
                "iterations": 1,
                "strength": strength
            },
            headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        if DEBUG:
            raise HTTPException(status_code=500, detail=f"Guard processing failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="画像保護の処理中にエラーが発生しました。しばらく経ってから再度お試しください。")
    finally:
        # レート制限ヘッダー用の情報を取得 (エラー時も確実にヘッダーを返すため)
        # ただし、Enterprise APIの場合は常に-1
        if is_enterprise:
            remaining_after, limit_after = -1, -1
        else:
            _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin, limit_type="guard")


@app.post("/guard-stream")
async def guard_image_stream(
    request: Request,
    file: UploadFile = File(...),
    iterations: int = Form(default=30),
    strength: float = Form(default=0.6)
):
    """
    SSEストリーミング版: リアルタイム進捗付きでMoonKnight保護を実行
    """
    global moonknight_engine

    if not MOONKNIGHT_ENABLED:
        raise HTTPException(status_code=503, detail="MoonKnight module not available")

    # Enterprise/レート制限チェック
    enterprise_info = verify_enterprise_api_key(request)
    is_enterprise = enterprise_info is not None

    key, is_vip, is_admin = None, False, False # Initialize for finally block

    if is_enterprise:
        track_enterprise_usage(enterprise_info["api_key"], "/guard-stream")
    else:
        key, is_vip, is_admin = get_rate_limit_key(request)
        allowed, remaining, limit = check_rate_limit(key, is_vip, is_admin, limit_type="guard")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"上限（{limit}枚）に達しました。1時間ごとに回復します。"
            )
        # トークン消費
        increment_rate_limit(key, is_vip, is_admin, limit_type="guard")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await validate_file_size(file)
    filename = file.filename
    user_id = enterprise_info["api_key"] if is_enterprise else key

    async def generate():
        import queue
        import threading

        progress_queue = queue.Queue()
        result_holder = {"result": None, "error": None}

        def progress_callback(current, total):
            progress_queue.put({"type": "progress", "current": current, "total": total})

        def process_in_thread():
            global moonknight_engine
            try:
                start_time = time.time()

                image = Image.open(io.BytesIO(contents)).convert("RGB")
                original_image = image.copy()

                # TrustMark透かし埋め込み（alpha=1.15、FastProtect前）
                watermark_hash = None
                if TRUSTMARK_ENABLED:
                    global trustmark_encoder
                    if trustmark_encoder is None:
                        import trustmark as tm
                        trustmark_encoder = tm.TrustMark()

                    timestamp_str = datetime.utcnow().isoformat() + "Z"
                    image = embed_watermark(
                        trustmark_encoder,
                        image,
                        user_id,
                        timestamp_str,
                        alpha=1.15
                    )
                    watermark_hash = create_user_watermark_mapping(
                        user_id,
                        timestamp_str,
                        capacity=trustmark_encoder.schemaCapacity()
                    )

                # 初期化
                if moonknight_engine is None:
                    progress_callback(10, 100) # Loading
                    moonknight_engine = MoonKnightV3(
                        model_dir="/home/techne/aicheckers/models/fastprotect",
                        device=str(device),
                        use_adaptive=True
                    )

                progress_callback(30, 100) # Analyzing & Protecting

                # MoonKnight実行
                # progress_callbackを渡してリアルタイム進捗通知
                protected_image = moonknight_engine.poison(
                    image, 
                    strength=strength,
                    progress_callback=progress_callback
                )
                poisoned_image = protected_image # Alias

                # DINOv3埋め込み抽出（最終画像から）
                if TRUSTMARK_ENABLED and watermark_hash is not None:
                    embedding = extract_dinov3_embedding(poisoned_image)
                    save_patrol_embedding(user_id, embedding, watermark_hash)

                # SSIM計算
                from skimage.metrics import structural_similarity as ssim
                orig_arr = np.array(original_image)
                pois_arr = np.array(poisoned_image.resize(original_image.size))
                ssim_value = ssim(orig_arr, pois_arr, channel_axis=2, data_range=255)

                # Base64エンコード
                buffer = io.BytesIO()
                poisoned_image.save(buffer, format="PNG", quality=95)
                protected_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                processing_time = time.time() - start_time

                result_holder["result"] = {
                    "protected_image": protected_base64,
                    "processing_time": round(processing_time, 3),
                    "ssim": round(ssim_value, 4),
                    "filename": filename,
                    "protection_applied": "MoonKnight V3",
                    "iterations": 1,
                    "strength": strength
                }
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                progress_queue.put({"type": "done"})

        # バックグラウンドスレッドで処理開始
        thread = threading.Thread(target=process_in_thread)
        thread.start()

        # SSEイベントを送信（パディング追加でブラウザバッファをフラッシュ）
        # ブラウザは通常2-4KB程度バッファリングするため、短いメッセージは届かない
        padding = " " * 2048  # 2KBパディング

        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
                if msg["type"] == "progress":
                    # パディング付きでブラウザのバッファを即座にフラッシュ
                    yield f"data: {json.dumps(msg)}\n\n:{padding}\n\n"
                    await asyncio.sleep(0)
                elif msg["type"] == "done":
                    break
            except queue.Empty:
                # Keep-alive
                yield f": keepalive{padding}\n\n"
                await asyncio.sleep(0)

        thread.join()

        # 最終結果を送信
        if result_holder["error"]:
            yield f"data: {json.dumps({'type': 'error', 'message': result_holder['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'complete', **result_holder['result']})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8"
        }
    )


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

    Enterprise API: X-API-Keyヘッダーで認証（レート制限なし）
    """
    # Enterprise APIキー認証チェック（優先）
    enterprise_info = verify_enterprise_api_key(request)
    is_enterprise = enterprise_info is not None

    if is_enterprise:
        # Enterprise APIはレート制限なし
        remaining, limit = -1, -1
        track_enterprise_usage(enterprise_info["api_key"], "/analyze-url")
    else:
        # 通常ユーザーのレート制限
        key, is_vip, is_admin = get_rate_limit_key(request)
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
            if not is_enterprise:
                increment_rate_limit(key, is_vip, is_admin)
                _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)
            else:
                remaining_after, limit_after = -1, -1
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

        # レート制限カウント増加（Enterprise以外）
        if not is_enterprise:
            increment_rate_limit(key, is_vip, is_admin)
            _, remaining_after, limit_after = check_rate_limit(key, is_vip, is_admin)
        else:
            remaining_after, limit_after = -1, -1

        return JSONResponse(
            content=response_data,
            headers={"X-RateLimit-Limit": str(limit_after), "X-RateLimit-Remaining": str(remaining_after)}
        )

    except HTTPException:
        raise
    except Exception as e:
        if DEBUG:
            raise HTTPException(status_code=500, detail=f"URL解析失敗: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="URL解析中にエラーが発生しました。URLが正しいか確認してください。")


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
        if DEBUG:
            raise HTTPException(status_code=500, detail=f"決済セッションの作成に失敗しました: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="決済セッションの作成に失敗しました。しばらく経ってから再度お試しください。")
    except Exception as e:
        print(f"Checkout session error: {e}")
        if DEBUG:
            raise HTTPException(status_code=500, detail=f"予期しないエラー: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="決済セッションの作成に失敗しました。しばらく経ってから再度お試しください。")


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

        # VIPステータス・管理者確認（ADMIN_EMAILSも自動VIP/Admin）
        is_admin = email in ADMIN_EMAILS
        is_vip = email in vip_users or is_admin

        return RedirectResponse(
            f"{FRONTEND_URL}?auth=success&token={jwt_token}&name={name}&email={email}&is_vip={str(is_vip).lower()}&is_admin={str(is_admin).lower()}"
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

        # VIPステータス・管理者確認（Twitterユーザーはusernameで確認、ADMIN_EMAILSも自動VIP/Admin）
        is_admin = email in ADMIN_EMAILS
        is_vip = email in vip_users or f"@{username}" in vip_users or is_admin

        return RedirectResponse(
            f"{FRONTEND_URL}?auth=success&token={jwt_token}&name={name}&email={email}&is_vip={str(is_vip).lower()}&is_admin={str(is_admin).lower()}"
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
    jwt_name = payload.get("name", "")  # JWTに含まれる名前（マジックリンク用）
    jwt_is_admin = payload.get("is_admin", False)

    user = users_db.get(user_id, {})
    # ADMIN_EMAILSも自動的にVIP/管理者扱い
    is_admin = jwt_is_admin or (email in ADMIN_EMAILS if email else False)
    is_vip = email in vip_users or is_admin

    # 名前はDBから取得、なければJWTから取得
    name = user.get("name", "") or jwt_name

    return {
        "id": user_id,
        "email": email,
        "name": name,
        "provider": user.get("provider", ""),
        "is_vip": is_vip,
        "is_admin": is_admin
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

        # マジックリンククリックをログ出力
        print(f"[MAGIC LINK CLICKED] email={email}, name={name}, is_admin={is_admin}")

        # 通常のログイン用JWTを発行（有効期限はマジックリンクの期限を引き継ぐ）
        exp = payload.get("exp")
        user_id = f"magic_{email}"

        login_token = jwt.encode(
            {
                "sub": user_id,
                "email": email,
                "name": name,  # 名前も埋め込み
                "exp": exp,
                "is_admin": is_admin,  # 管理者フラグを埋め込み
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM
        )

        # フロントエンドにリダイレクト（トークンをクエリパラメータで渡す）
        is_admin_str = "true" if is_admin else "false"
        redirect_url = f"https://aicheckers.net?magic_token={login_token}&name={name}&email={email}&is_vip=true&is_admin={is_admin_str}"
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


# ==================== Enterprise API管理 ====================

class CreateEnterpriseKeyRequest(BaseModel):
    company_name: str
    contact_email: str
    plan: str = "standard"  # standard, unlimited
    expires_days: int = 365  # デフォルト1年


@app.post("/admin/enterprise/create-key")
async def create_enterprise_key(request: Request, body: CreateEnterpriseKeyRequest):
    """Enterprise APIキーを発行（管理者のみ）"""
    # 管理者認証
    key, is_vip, is_admin = get_rate_limit_key(request)
    if not is_admin:
        raise HTTPException(status_code=403, detail="管理者権限が必要です")

    global enterprise_keys

    # APIキー生成
    api_key = generate_api_key()
    created_at = datetime.now()
    expires_at = created_at + timedelta(days=body.expires_days) if body.expires_days > 0 else None

    key_info = {
        "company_name": body.company_name,
        "contact_email": body.contact_email,
        "plan": body.plan,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat() if expires_at else None,
        "revoked": False,
    }

    enterprise_keys[api_key] = key_info
    save_enterprise_keys(enterprise_keys)

    print(f"Enterprise key created for: {body.company_name} ({body.contact_email})")

    return {
        "api_key": api_key,
        **key_info
    }


@app.get("/admin/enterprise/list-keys")
async def list_enterprise_keys(request: Request):
    """発行済みEnterprise APIキー一覧（管理者のみ）"""
    key, is_vip, is_admin = get_rate_limit_key(request)
    if not is_admin:
        raise HTTPException(status_code=403, detail="管理者権限が必要です")

    result = []
    for api_key, info in enterprise_keys.items():
        # APIキーは先頭と末尾のみ表示
        masked_key = api_key[:20] + "..." + api_key[-8:]
        result.append({
            "api_key_masked": masked_key,
            "api_key_full": api_key,  # 管理者には全体を表示
            **info
        })

    return {"keys": result}


@app.post("/admin/enterprise/revoke-key/{api_key}")
async def revoke_enterprise_key(request: Request, api_key: str):
    """Enterprise APIキーを無効化（管理者のみ）"""
    key, is_vip, is_admin = get_rate_limit_key(request)
    if not is_admin:
        raise HTTPException(status_code=403, detail="管理者権限が必要です")

    if api_key not in enterprise_keys:
        raise HTTPException(status_code=404, detail="APIキーが見つかりません")

    enterprise_keys[api_key]["revoked"] = True
    enterprise_keys[api_key]["revoked_at"] = datetime.now().isoformat()
    save_enterprise_keys(enterprise_keys)

    return {"status": "revoked", "api_key": api_key}


@app.get("/admin/enterprise/usage/{api_key}")
async def get_enterprise_usage(request: Request, api_key: str, month: str = None):
    """Enterprise APIキーの使用量を取得（管理者のみ）"""
    key, is_vip, is_admin = get_rate_limit_key(request)
    if not is_admin:
        raise HTTPException(status_code=403, detail="管理者権限が必要です")

    if api_key not in enterprise_keys:
        raise HTTPException(status_code=404, detail="APIキーが見つかりません")

    if month is None:
        month = datetime.now().strftime("%Y-%m")

    usage = enterprise_usage.get(api_key, {}).get(month, {"total": 0, "endpoints": {}})

    return {
        "api_key": api_key[:20] + "..." + api_key[-8:],
        "company_name": enterprise_keys[api_key].get("company_name"),
        "month": month,
        "usage": usage
    }


@app.get("/admin/enterprise/usage-all")
async def get_all_enterprise_usage(request: Request, month: str = None):
    """全Enterprise APIキーの使用量サマリー（管理者のみ）"""
    key, is_vip, is_admin = get_rate_limit_key(request)
    if not is_admin:
        raise HTTPException(status_code=403, detail="管理者権限が必要です")

    if month is None:
        month = datetime.now().strftime("%Y-%m")

    result = []
    for api_key, info in enterprise_keys.items():
        usage = enterprise_usage.get(api_key, {}).get(month, {"total": 0, "endpoints": {}})
        result.append({
            "api_key_masked": api_key[:20] + "..." + api_key[-8:],
            "company_name": info.get("company_name"),
            "plan": info.get("plan"),
            "revoked": info.get("revoked", False),
            "month": month,
            "total_requests": usage.get("total", 0),
        })

    # 使用量順にソート
    result.sort(key=lambda x: x["total_requests"], reverse=True)

    return {"month": month, "usage": result}


@app.get("/enterprise/verify")
async def verify_enterprise_key_endpoint(request: Request):
    """Enterprise APIキーの有効性を確認（企業向け）"""
    enterprise_info = verify_enterprise_api_key(request)
    if not enterprise_info:
        raise HTTPException(status_code=401, detail="無効なAPIキーです")

    # 使用量を取得
    api_key = enterprise_info["api_key"]
    month = datetime.now().strftime("%Y-%m")
    usage = enterprise_usage.get(api_key, {}).get(month, {"total": 0, "endpoints": {}})

    return {
        "valid": True,
        "company_name": enterprise_info.get("company_name"),
        "plan": enterprise_info.get("plan"),
        "expires_at": enterprise_info.get("expires_at"),
        "current_month_usage": usage.get("total", 0)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
