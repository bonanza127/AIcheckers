"""
AIcheckers Backend API
AniXplore (Modal) をメイン、legekka をフォールバックとして使用
"""

import io
import time
import base64
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
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

MODEL_NAME = "legekka/AI-Anime-Image-Detector-ViT"


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
    """起動時にlegekkaモデルをロード（フォールバック用）"""
    global model, processor, device

    print(f"Loading fallback model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager"  # Attention出力を有効にするため
    )
    model.to(device)
    model.eval()

    print("Fallback model loaded successfully!")
    
    # AniXplore (Modal) の状態確認
    if MODAL_AVAILABLE:
        print("Modal available, AniXplore will be used as primary detector")
    else:
        print("Modal not available, using legekka as primary detector")
    
    yield

    # クリーンアップ
    del model, processor
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


@app.get("/")
async def root():
    return {
        "status": "ok", 
        "primary_model": "AniXplore (Modal)" if MODAL_AVAILABLE else MODEL_NAME,
        "fallback_model": MODEL_NAME
    }


@app.get("/health")
async def health_check():
    # 軽量なヘルスチェック（Modal接続は試みない）
    return {
        "status": "healthy",
        "primary_model": "AniXplore (Modal)",
        "primary_status": "available" if MODAL_AVAILABLE else "unavailable",
        "fallback_model": MODEL_NAME,
        "fallback_loaded": model is not None,
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


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Form(default="anixplore")  # "anixplore" or "legekka"
):
    """
    画像を解析してAI生成かどうかを判定

    Args:
        model: 使用するモデル ("anixplore" or "legekka")

    Returns:
        - is_ai: AI生成かどうか
        - ai_score: AI生成の確信度 (0-100)
        - human_score: 人間作の確信度 (0-100)
        - processing_time: 処理時間(秒)
        - model_used: 使用したモデル
        - attention_map: Attention Mapのヒートマップ (legekka使用時のみ)
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
        use_anixplore = model == "anixplore" and MODAL_AVAILABLE

        if use_anixplore:
            try:
                result = await analyze_with_anixplore(contents)
            except Exception as e:
                error_info = f"AniXplore failed: {str(e)}"
                print(error_info)
        
        # 2. フォールバック: legekka (ローカル)
        if result is None:
            try:
                result = await analyze_with_legekka(image, contents)
                if error_info:
                    result["fallback_reason"] = error_info
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Both models failed. Last error: {str(e)}")

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
            "fallback_reason": result.get("fallback_reason")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
