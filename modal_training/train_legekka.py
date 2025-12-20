"""
legekka ファインチューニング - Modal T4 GPU
AnimeDL-2Mデータセットを使用してViTモデルをファインチューニング
"""

import modal
import os
from pathlib import Path

# Modal設定
app = modal.App("legekka-finetune")

# 学習データ用Volume
training_volume = modal.Volume.from_name("legekka-training-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("legekka-checkpoints", create_if_missing=True)

# Docker image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "accelerate>=0.24.0",
        "evaluate>=0.4.0",
        "gdown>=4.7.0",
        "tqdm>=4.66.0",
    )
)

# 学習設定
TRAIN_CONFIG = {
    "model_name": "legekka/AI-Anime-Image-Detector-ViT",
    "learning_rate": 2e-4,
    "batch_size": 16,  # 小規模検証用
    "epochs": 1,       # 小規模検証用
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "eval_steps": 50,  # 小規模検証用
    "save_steps": 100,
    "logging_steps": 10,
    "max_samples": 1000,  # 小規模検証用
}

# 本番設定
PROD_CONFIG = {
    "model_name": "legekka/AI-Anime-Image-Detector-ViT",
    "learning_rate": 2e-4,
    "batch_size": 32,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "eval_steps": 100,
    "save_steps": 500,
    "logging_steps": 50,
    "max_samples": None,  # 全データ使用
}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,  # 4時間
    volumes={
        "/data": training_volume,
        "/checkpoints": checkpoints_volume,
    },
)
def train(config: dict = None, is_validation: bool = True):
    """ファインチューニング実行"""
    import torch
    from transformers import (
        AutoModelForImageClassification,
        AutoImageProcessor,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset, DatasetDict
    from PIL import Image
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    cfg = config or (TRAIN_CONFIG if is_validation else PROD_CONFIG)
    print(f"=== Training Configuration ===")
    print(f"Mode: {'Validation' if is_validation else 'Production'}")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # モデルとプロセッサーをロード
    print(f"\nLoading model: {cfg['model_name']}")
    processor = AutoImageProcessor.from_pretrained(cfg["model_name"])
    model = AutoModelForImageClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )

    # データ読み込み
    print("\nLoading training data...")
    data_dir = Path("/data")

    def load_images_from_dir(directory: Path, label: int, max_samples: int = None):
        """ディレクトリから画像を読み込む"""
        images = []
        labels = []

        if not directory.exists():
            print(f"Warning: {directory} does not exist")
            return images, labels

        files = list(directory.glob("*.jpg")) + list(directory.glob("*.png")) + list(directory.glob("*.webp"))
        if max_samples:
            files = files[:max_samples]

        for img_path in tqdm(files, desc=f"Loading {directory.name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        return images, labels

    # AI画像（label=1）とリアル画像（label=0）を読み込む
    max_per_class = cfg["max_samples"] // 2 if cfg["max_samples"] else None

    ai_images, ai_labels = load_images_from_dir(data_dir / "ai", 1, max_per_class)
    real_images, real_labels = load_images_from_dir(data_dir / "real", 0, max_per_class)

    print(f"Loaded {len(ai_images)} AI images, {len(real_images)} real images")

    if len(ai_images) == 0 or len(real_images) == 0:
        print("ERROR: No training data found!")
        print("Please upload data to the volume first using upload_training_data()")
        return {"error": "No training data"}

    # データセット作成
    all_images = ai_images + real_images
    all_labels = ai_labels + real_labels

    # シャッフル
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # train/val split (90/10)
    split_idx = int(len(all_images) * 0.9)

    def preprocess(examples):
        """画像を前処理"""
        inputs = processor(images=examples["image"], return_tensors="pt")
        inputs["labels"] = torch.tensor(examples["label"])
        return inputs

    train_dataset = Dataset.from_dict({
        "image": all_images[:split_idx],
        "label": all_labels[:split_idx],
    })

    val_dataset = Dataset.from_dict({
        "image": all_images[split_idx:],
        "label": all_labels[split_idx:],
    })

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 前処理適用
    def transform(examples):
        inputs = processor(images=examples["image"], return_tensors="pt")
        inputs["labels"] = torch.tensor(examples["label"])
        return {k: v.squeeze(0) if v.dim() > 0 else v for k, v in inputs.items()}

    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)

    # 評価関数
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="binary"),
        }

    # トレーニング引数
    output_dir = "/checkpoints/legekka-finetuned"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
        fp16=True,  # Mixed precision for T4
    )

    # Trainer初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # トレーニング開始
    print("\n=== Starting Training ===")
    train_result = trainer.train()

    # 結果保存
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Volume同期
    checkpoints_volume.commit()

    # 最終評価
    eval_result = trainer.evaluate()

    print("\n=== Training Complete ===")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Eval accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"Eval F1: {eval_result['eval_f1']:.4f}")

    return {
        "train_loss": train_result.training_loss,
        "eval_accuracy": eval_result["eval_accuracy"],
        "eval_f1": eval_result["eval_f1"],
        "output_dir": output_dir,
    }


@app.function(
    image=image,
    volumes={"/data": training_volume},
    timeout=3600,
)
def upload_training_data(local_ai_dir: str, local_real_dir: str):
    """ローカルからトレーニングデータをアップロード"""
    import shutil
    from pathlib import Path

    data_dir = Path("/data")
    ai_dir = data_dir / "ai"
    real_dir = data_dir / "real"

    ai_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # コピー
    for src in Path(local_ai_dir).glob("*"):
        if src.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            shutil.copy(src, ai_dir / src.name)

    for src in Path(local_real_dir).glob("*"):
        if src.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            shutil.copy(src, real_dir / src.name)

    training_volume.commit()

    ai_count = len(list(ai_dir.glob("*")))
    real_count = len(list(real_dir.glob("*")))

    return {"ai_images": ai_count, "real_images": real_count}


@app.function(
    image=image,
    volumes={"/data": training_volume},
    timeout=7200,  # 2時間
)
def download_animedl2m(subset: str = "Illustrious", max_images: int = 5000):
    """AnimeDL-2MからデータをダウンロードしてVolume に保存"""
    import gdown
    import zipfile
    import shutil
    from pathlib import Path
    from tqdm import tqdm

    data_dir = Path("/data")
    temp_dir = data_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # AnimeDL-2M Google Drive IDs (実際のIDに置き換え必要)
    # https://github.com/FlyTweety/AnimeDL2M
    DRIVE_IDS = {
        "Illustrious": "1XXXXXXXXX",  # 要実ID
        "Pony": "1XXXXXXXXX",
        "Other": "1XXXXXXXXX",
        "Real": "1XXXXXXXXX",
    }

    if subset not in DRIVE_IDS:
        return {"error": f"Unknown subset: {subset}. Available: {list(DRIVE_IDS.keys())}"}

    print(f"Downloading {subset} subset...")
    # gdown.download(f"https://drive.google.com/uc?id={DRIVE_IDS[subset]}", ...)

    # TODO: 実装完了後にアンコメント

    return {"status": "download_function_ready", "subset": subset}


@app.function(
    image=image,
    gpu="T4",
    volumes={"/checkpoints": checkpoints_volume},
    timeout=600,
)
def test_model(image_bytes: bytes):
    """ファインチューニング済みモデルでテスト"""
    import torch
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    import io

    model_dir = "/checkpoints/legekka-finetuned"

    if not Path(model_dir).exists():
        return {"error": "Model not found. Run training first."}

    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 画像処理
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    human_score = float(probs[0]) * 100
    ai_score = float(probs[1]) * 100

    return {
        "is_ai": ai_score > human_score,
        "ai_score": round(ai_score, 2),
        "human_score": round(human_score, 2),
        "confidence": round(max(ai_score, human_score), 2),
    }


@app.local_entrypoint()
def main(mode: str = "validate", local_data: str = None):
    """
    メインエントリーポイント

    Usage:
        modal run train_legekka.py --mode validate   # 小規模検証
        modal run train_legekka.py --mode train      # 本番トレーニング
        modal run train_legekka.py --mode test --local-data /path/to/image.png
    """
    print(f"legekka Fine-tuning - Mode: {mode}")

    if mode == "validate":
        result = train.remote(is_validation=True)
        print(f"Validation result: {result}")

    elif mode == "train":
        result = train.remote(is_validation=False)
        print(f"Training result: {result}")

    elif mode == "test":
        if not local_data:
            print("Error: --local-data required for test mode")
            return
        with open(local_data, "rb") as f:
            image_bytes = f.read()
        result = test_model.remote(image_bytes)
        print(f"Test result: {result}")

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: validate, train, test")
