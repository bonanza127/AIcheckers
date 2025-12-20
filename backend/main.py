"""
AIcheckers Backend API
AniXplore (Modal) をメイン、legekka をフォールバック、DINOv3 (ローカル) をオプションとして使用
"""

import io
import time
import base64
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoModel
import matplotlib
matplotlib.use('Agg')  # GUIなしで使用
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Modal (AniXplore用)
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    print("Warning: modal not installed, AniXplore will not be available")


# グローバル変数
model = None  # legekka (フォールバック用)
processor = None
device = None

# DINOv3 ローカル用
dinov3_model = None
dinov3_processor = None
dinov3_classifier = None
model_centroids = {}  # モデル別セントロイド（類似度計算用）

MODEL_NAME = "legekka/AI-Anime-Image-Detector-ViT"
DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_CLASSIFIER_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
HF_TOKEN = "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"


def get_anixplore_detector():
    """AniXplore detector インスタンスを取得"""
    if not MODAL_AVAILABLE:
        return None
    try:
        Detector = modal.Cls.from_name("anixplore-detector", "AniXploreDetector")
        return Detector()
    except Exception as e:
        print(f"Failed to get AniXplore detector: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にモデルをロード"""
    global model, processor, device
    global dinov3_model, dinov3_processor, dinov3_classifier, model_centroids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # legekka ロード
    print(f"Loading legekka model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    print("legekka model loaded!")

    # DINOv3 ローカルロード
    print(f"Loading DINOv3 model: {DINOV3_MODEL_NAME}")
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        dinov3_processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME, token=HF_TOKEN)
        dinov3_model = AutoModel.from_pretrained(
            DINOV3_MODEL_NAME,
            token=HF_TOKEN,
            attn_implementation="eager"  # attention出力に必要
        )
        dinov3_model.to(device)
        dinov3_model.eval()
        
        # 分類器ロード
        if DINOV3_CLASSIFIER_PATH.exists():
            checkpoint = torch.load(DINOV3_CLASSIFIER_PATH, map_location=device)
            dinov3_classifier = nn.Linear(768, 2).to(device)
            dinov3_classifier.load_state_dict(checkpoint["classifier"])
            dinov3_classifier.eval()
            print(f"DINOv3 classifier loaded! (val_acc: {checkpoint.get('val_acc', 'N/A')})")
        else:
            print(f"Warning: DINOv3 classifier not found at {DINOV3_CLASSIFIER_PATH}")
            
        print("DINOv3 model loaded!")
    except Exception as e:
        print(f"Failed to load DINOv3: {e}")
        dinov3_model = None

    # モデル別セントロイドをロード（類似度計算用）
    print("Loading model centroids for similarity detection...")
    model_name_map = {
        "illustrious_ai": "Illustrious",
        "pony_ai": "Pony",
        "sdxl10_ai": "SDXL 1.0",
        "sd15_ai": "SD 1.5",
        "flux1d_ai": "FLUX.1",
        "other_ai": "Other",
    }
    for npy_file in EMBEDDINGS_DIR.glob("*_ai.npy"):
        key = npy_file.stem  # e.g., "illustrious_ai"
        embeddings = np.load(npy_file)
        centroid = embeddings.mean(axis=0)
        # 正規化（コサイン類似度用）
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        display_name = model_name_map.get(key, key.replace("_ai", "").title())
        model_centroids[display_name] = centroid
        print(f"  Loaded centroid: {display_name} ({len(embeddings)} samples)")
    print(f"Loaded {len(model_centroids)} model centroids")

    # AniXplore (Modal) の状態確認
    if MODAL_AVAILABLE:
        print("Modal available, AniXplore will be used as primary detector")
    
    yield

    # クリーンアップ
    del model, processor, dinov3_model, dinov3_processor, dinov3_classifier
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
)


def generate_attention_heatmap(attentions, original_image):
    """
    Attention weightsからヒートマップを生成
    
    Args:
        attentions: モデルのattention outputs
        original_image: 元のPIL Image
    
    Returns:
        Base64エンコードされたヒートマップ画像
    """
    # 最後の層のattentionを取得 (shape: [batch, heads, seq_len, seq_len])
    last_attention = attentions[-1]
    
    # 全ヘッドの平均を取る
    attention_avg = last_attention.mean(dim=1)[0]  # [seq_len, seq_len]
    
    # CLSトークン（index 0）から各パッチへのattentionを取得
    cls_attention = attention_avg[0, 1:]  # CLSトークン自身を除く
    
    # 14x14のグリッドにリシェイプ（ViT-base: 224/16 = 14）
    num_patches = int(np.sqrt(cls_attention.shape[0]))
    attention_map = cls_attention.reshape(num_patches, num_patches).cpu().numpy()
    
    # 正規化
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 元の画像サイズにリサイズ
    original_size = original_image.size  # (width, height)
    attention_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
        original_size, Image.BILINEAR
    )) / 255.0
    
    # ヒートマップを作成（元画像にオーバーレイ）
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(original_image)
    ax.imshow(attention_resized, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Base64エンコード
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_dinov3_attention_heatmap(model, inputs, original_image, device):
    """
    DINOv3のself-attentionからヒートマップを生成
    DINOv3は自己教師学習で訓練されているため、セマンティックに意味のある領域を強調する
    
    Args:
        model: DINOv3モデル
        inputs: 前処理済み入力
        original_image: 元のPIL Image
        device: torch device
    
    Returns:
        Base64エンコードされたヒートマップ画像
    """
    # Attention出力を取得するためにフックを設定
    attentions = []
    
    def get_attention_hook(module, input, output):
        # DINOv3のattention layerからattention weightsを取得
        attentions.append(output[1] if isinstance(output, tuple) else output)
    
    # 最後のattention layerにフックを登録
    # DINOv3 (dinov2ベース) のアーキテクチャに合わせる
    hook_handle = None
    try:
        # DINOv3のViTエンコーダの最後の層のattentionを取得
        encoder_layers = model.encoder.layer
        last_layer = encoder_layers[-1]
        hook_handle = last_layer.attention.attention.register_forward_hook(
            lambda m, i, o: attentions.append(o[1]) if len(o) > 1 else None
        )
    except AttributeError:
        # 別のアーキテクチャの場合
        pass
    
    # output_attentions=Trueで推論
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    if hook_handle:
        hook_handle.remove()
    
    # attentionsを取得
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        model_attentions = outputs.attentions
    else:
        return None
    
    # 最後の層のattentionを取得 (shape: [batch, heads, seq_len, seq_len])
    last_attention = model_attentions[-1]
    
    # 全ヘッドの平均を取る
    attention_avg = last_attention.mean(dim=1)[0]  # [seq_len, seq_len]
    
    # CLSトークン（index 0）から各パッチへのattentionを取得
    cls_attention = attention_avg[0, 1:]  # CLSトークン自身を除く
    
    # パッチ数からグリッドサイズを計算（DINOv3: 518/14 = 37 または 224/14 = 16）
    num_patches = int(np.sqrt(cls_attention.shape[0]))
    if num_patches * num_patches != cls_attention.shape[0]:
        # 正方形でない場合はスキップ
        return None
    
    attention_map = cls_attention.reshape(num_patches, num_patches).cpu().numpy()
    
    # 正規化
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 元の画像サイズにリサイズ
    original_size = original_image.size  # (width, height)
    attention_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
        original_size, Image.BILINEAR
    )) / 255.0
    
    # ヒートマップを作成（元画像にオーバーレイ）
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(original_image)
    ax.imshow(attention_resized, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Base64エンコード
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.get("/")
async def root():
    return {
        "status": "ok",
        "primary_model": "AniXplore (Modal)" if MODAL_AVAILABLE else MODEL_NAME,
        "fallback_model": MODEL_NAME,
        "optional_models": ["dinov3", "legekka"]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "primary_model": "AniXplore (Modal)",
        "primary_status": "available" if MODAL_AVAILABLE else "unavailable",
        "fallback_model": MODEL_NAME,
        "fallback_loaded": model is not None,
        "dinov3_status": "available" if dinov3_model is not None and dinov3_classifier is not None else "unavailable",
        "dinov3_local": True,
        "device": str(device) if device else None,
        "cuda_available": torch.cuda.is_available()
    }


async def analyze_with_anixplore(image_bytes: bytes) -> dict:
    """AniXplore (Modal) で解析"""
    detector = get_anixplore_detector()
    if not detector:
        raise Exception("AniXplore detector not available")
    
    result = detector.detect.remote(image_bytes)
    
    # AniXploreの結果をAPIレスポンス形式に変換
    ai_score = result["probability"] * 100
    human_score = 100 - ai_score
    
    return {
        "ai_score": round(ai_score, 2),
        "human_score": round(human_score, 2),
        "confidence": round(result["confidence"] * 100, 2),
        "is_ai": result["is_ai"],
        "model_used": "AniXplore",
        "attention_map": None  # AniXploreは周波数分析ベースなのでattention mapなし
    }


async def analyze_with_legekka(image: Image.Image, image_bytes: bytes) -> dict:
    """legekka (ローカル) で解析"""
    if model is None:
        raise Exception("Legekka model not loaded")

    # 前処理
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推論（attention出力も取得）
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        attentions = outputs.attentions

    # Attention Mapヒートマップを生成
    attention_heatmap = generate_attention_heatmap(attentions, image)

    # ラベルマッピング
    id2label = model.config.id2label

    # スコア取得
    scores = {}
    for idx, label in id2label.items():
        scores[label.lower()] = float(probabilities[idx].cpu()) * 100

    # AI/Humanのスコアを正規化
    ai_score = scores.get("ai", scores.get("artificial", 0))
    human_score = scores.get("human", scores.get("real", 100 - ai_score))

    return {
        "ai_score": round(ai_score, 2),
        "human_score": round(human_score, 2),
        "confidence": round(max(ai_score, human_score), 2),
        "is_ai": ai_score > human_score,
        "model_used": "legekka",
        "attention_map": attention_heatmap
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
        "model_used": "DINOv3",
        "attention_map": attention_heatmap,
        "forensic_logs": forensic_logs,
        "detected_traces": detected_traces,
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Form(default="anixplore")  # "anixplore", "legekka", or "dinov3"
):
    """
    画像を解析してAI生成かどうかを判定

    Args:
        model: 使用するモデル ("anixplore", "legekka", or "dinov3")

    Returns:
        - is_ai: AI生成かどうか
        - ai_score: AI生成の確信度 (0-100)
        - human_score: 人間作の確信度 (0-100)
        - processing_time: 処理時間(秒)
        - model_used: 使用したモデル
        - attention_map: Attention Mapのヒートマップ (legekka/DINOv3使用時)
    """
    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        start_time = time.time()

        # 画像読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = None
        error_info = None

        # ユーザー指定のモデルを使用
        if model == "dinov3":
            try:
                result = await analyze_with_dinov3(image)
            except Exception as e:
                error_info = f"DINOv3 failed: {str(e)}"
                print(error_info)
        elif model == "anixplore" and MODAL_AVAILABLE:
            try:
                result = await analyze_with_anixplore(contents)
            except Exception as e:
                error_info = f"AniXplore failed: {str(e)}"
                print(error_info)
        elif model == "legekka":
            try:
                result = await analyze_with_legekka(image, contents)
            except Exception as e:
                error_info = f"Legekka failed: {str(e)}"
                print(error_info)
        
        # フォールバック: legekka (ローカル)
        if result is None:
            try:
                result = await analyze_with_legekka(image, contents)
                if error_info:
                    result["fallback_reason"] = error_info
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"All models failed. Last error: {str(e)}")

        processing_time = time.time() - start_time

        return {
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
            "fallback_reason": result.get("fallback_reason")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
