"""
AIcheckers Backend API
legekka/AI-Anime-Image-Detector-ViT を使用したAI画像判定API
"""

import io
import time
import base64
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import matplotlib
matplotlib.use('Agg')  # GUIなしで使用
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# グローバル変数
model = None
processor = None
device = None

MODEL_NAME = "legekka/AI-Anime-Image-Detector-ViT"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にモデルをロード"""
    global model, processor, device

    print(f"Loading model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager"  # Attention出力を有効にするため
    )
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
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
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://aicheckers.net"],
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
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    画像を解析してAI生成かどうかを判定

    Returns:
        - is_ai: AI生成かどうか
        - ai_score: AI生成の確信度 (0-100)
        - human_score: 人間作の確信度 (0-100)
        - processing_time: 処理時間(秒)
        - attention_map: Attention Mapのヒートマップ (Base64)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ファイル形式チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        start_time = time.time()

        # 画像読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

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

        processing_time = time.time() - start_time

        # ラベルマッピング（legekkaモデルの場合）
        id2label = model.config.id2label

        # スコア取得
        scores = {}
        for idx, label in id2label.items():
            scores[label.lower()] = float(probabilities[idx].cpu()) * 100

        # AI/Humanのスコアを正規化
        ai_score = scores.get("ai", scores.get("artificial", 0))
        human_score = scores.get("human", scores.get("real", 100 - ai_score))

        is_ai = ai_score > human_score

        return {
            "is_ai": is_ai,
            "ai_score": round(ai_score, 2),
            "human_score": round(human_score, 2),
            "confidence": round(max(ai_score, human_score), 2),
            "verdict": "AI DETECTED" if is_ai else "HUMAN CONFIRMED",
            "processing_time": round(processing_time, 3),
            "filename": file.filename,
            "attention_map": attention_heatmap
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
