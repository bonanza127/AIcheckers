#!/usr/bin/env python3
"""
Human Signature Detection Module for AIcheckers

秘密鍵ベースの署名を検出し、画像が「Human Verified」かどうかを判定する。
Ironclad v3.1で埋め込まれた署名を検出。

使用方法:
    from lib.signature import detect_human_signature

    result = detect_human_signature(image_bytes_or_pil)
    if result["detected"]:
        print("Human Verified!")
"""

import os
import hashlib
import hmac
from datetime import datetime
from typing import Union, Optional

import numpy as np
from PIL import Image

# 環境変数から秘密鍵を取得（デフォルト値あり）
SIGNATURE_SECRET_KEY = os.getenv("SIGNATURE_SECRET_KEY", "AICHECKERS_DEFAULT_KEY")
SIGNATURE_VERSION = os.getenv("SIGNATURE_VERSION", None)  # Noneなら現在月
DETECTION_THRESHOLD = float(os.getenv("SIGNATURE_THRESHOLD", "0.20"))

# 遅延インポート用
_torch = None
_ptwt = None
_kornia = None
_imagehash = None


def _lazy_imports():
    """GPU/CPUどちらでも動作するように遅延インポート"""
    global _torch, _ptwt, _kornia, _imagehash
    if _torch is None:
        import torch
        import ptwt
        import kornia
        import imagehash
        _torch = torch
        _ptwt = ptwt
        _kornia = kornia
        _imagehash = imagehash


def detect_human_signature(
    image: Union[bytes, Image.Image, np.ndarray],
    secret_key: str = None,
    version: str = None,
    normalize_resolution: bool = True,
    canonical_size: int = 512,
    threshold: float = None,
) -> dict:
    """
    画像から Human Signature を検出

    Args:
        image: 画像データ（bytes, PIL.Image, またはnumpy array）
        secret_key: 署名生成に使用した秘密鍵（Noneなら環境変数）
        version: 署名バージョン（Noneなら現在月）
        normalize_resolution: 解像度正規化を行うか
        canonical_size: 正規化サイズ
        threshold: 検出閾値（Noneならデフォルト）

    Returns:
        dict: {
            "detected": bool,           # 署名が検出されたか
            "correlation": float,       # 相関係数
            "correlation_lh": float,    # LH成分の相関
            "correlation_hl": float,    # HL成分の相関
            "threshold": float,         # 使用した閾値
            "version": str,             # 署名バージョン
            "image_salt": str,          # 画像のpHash
        }
    """
    _lazy_imports()

    torch = _torch
    ptwt = _ptwt
    kornia = _kornia
    imagehash = _imagehash
    import torch.nn.functional as F

    # パラメータ設定
    key = secret_key or SIGNATURE_SECRET_KEY
    ver = version or SIGNATURE_VERSION or datetime.now().strftime("%Y%m")
    thresh = threshold if threshold is not None else DETECTION_THRESHOLD
    full_key = f"{key}_{ver}".encode()

    # デバイス選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 画像をテンソルに変換
    if isinstance(image, bytes):
        import io
        img_pil = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    arr = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 解像度処理
    _, _, h, w = img_tensor.shape
    if normalize_resolution:
        scale = canonical_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        img_processed = F.interpolate(
            img_tensor, (new_h, new_w), mode='bilinear', align_corners=False
        )
    else:
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            img_processed = F.interpolate(
                img_tensor, (new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            img_processed = img_tensor

    # YCbCr変換
    img_ycbcr = kornia.color.rgb_to_ycbcr(img_processed)
    Y = img_ycbcr[:, 0:1, :, :]

    # pHashベースソルト
    Y_np = Y.squeeze(0).squeeze(0).cpu().numpy()
    Y_pil = Image.fromarray((Y_np * 255).clip(0, 255).astype(np.uint8), mode='L')
    phash = imagehash.phash(Y_pil, hash_size=8)
    image_salt = str(phash)

    # DWT分解
    WAVELET = 'bior1.3'
    coeffs = ptwt.wavedec2(Y, WAVELET, level=1)
    LH, HL = coeffs[1][0], coeffs[1][1]

    # 期待されるパターン生成
    hmac_key = hmac.new(full_key, image_salt.encode(), hashlib.sha256).digest()
    seed = int.from_bytes(hmac_key[:8], 'big')
    rng = torch.Generator().manual_seed(seed)
    expected = (torch.rand(LH.shape, generator=rng) * 2 - 1).to(device)

    # 相関係数計算
    def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        a_norm = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-8)
        b_norm = (b_flat - b_flat.mean()) / (b_flat.std() + 1e-8)
        return (a_norm * b_norm).mean().item()

    corr_lh = correlation(LH, expected)
    corr_hl = correlation(HL, expected)
    avg_corr = (corr_lh + corr_hl) / 2

    return {
        "detected": avg_corr > thresh,
        "correlation": avg_corr,
        "correlation_lh": corr_lh,
        "correlation_hl": corr_hl,
        "threshold": thresh,
        "version": ver,
        "image_salt": image_salt,
    }


def check_multiple_versions(
    image: Union[bytes, Image.Image, np.ndarray],
    secret_key: str = None,
    versions: list = None,
    **kwargs
) -> dict:
    """
    複数のバージョンで署名検出を試みる

    古いバージョンで署名された画像も検出できるようにする
    """
    if versions is None:
        # 過去3ヶ月分をチェック
        from datetime import datetime, timedelta
        now = datetime.now()
        versions = [
            (now - timedelta(days=30 * i)).strftime("%Y%m")
            for i in range(3)
        ]

    best_result = None
    best_corr = -1.0

    for ver in versions:
        result = detect_human_signature(
            image, secret_key=secret_key, version=ver, **kwargs
        )
        if result["correlation"] > best_corr:
            best_corr = result["correlation"]
            best_result = result

        if result["detected"]:
            return result

    return best_result


# テスト用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python signature.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path)

    print(f"Checking signature for: {image_path}")
    result = detect_human_signature(img)

    print(f"\nResults:")
    print(f"  Detected: {result['detected']}")
    print(f"  Correlation: {result['correlation']:.4f}")
    print(f"  LH: {result['correlation_lh']:.4f}")
    print(f"  HL: {result['correlation_hl']:.4f}")
    print(f"  Threshold: {result['threshold']}")
    print(f"  Version: {result['version']}")
    print(f"  Image salt: {result['image_salt']}")
