"""
AIcheckers Backend API
legekka/AI-Anime-Image-Detector-ViT を使用したAI画像判定API
"""

import io
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor


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
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
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
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        processing_time = time.time() - start_time

        # ラベルマッピング（legekkaモデルの場合）
        # id2label: {0: "ai", 1: "human"} or similar
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
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
