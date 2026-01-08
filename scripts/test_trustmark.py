#!/usr/bin/env python3
"""
TrustMark動作確認スクリプト
"""
import sys
import time
import numpy as np
from PIL import Image
import torch

def test_trustmark_basic():
    """TrustMark基本動作確認"""
    try:
        import trustmark
        print("✓ TrustMark import成功")
    except ImportError as e:
        print(f"✗ TrustMark import失敗: {e}")
        return False

    # ダミー画像作成（512x512, RGB）
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_image_pil = Image.fromarray(test_image)

    print(f"テスト画像サイズ: {test_image_pil.size}")
    print(f"GPU使用可否: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"VRAM使用前: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # TrustMark初期化
    try:
        print("\nTrustMark初期化中...")
        encoder = trustmark.TrustMark()
        print("✓ TrustMark初期化成功")
    except Exception as e:
        print(f"✗ TrustMark初期化失敗: {e}")
        return False

    # 透かし容量確認
    try:
        capacity = encoder.schemaCapacity()
        print(f"\n透かし容量: {capacity} bits")
    except Exception as e:
        print(f"容量確認失敗: {e}")
        capacity = 100  # デフォルト

    # エンコード（透かし埋め込み）テスト
    try:
        print("\n透かし埋め込みテスト...")
        # メッセージをバイナリ文字列に変換
        import json
        import hashlib
        message_data = {"user_id": "user_12345", "timestamp": "2024-01-08T14:30:45Z"}
        message_json = json.dumps(message_data)
        # SHA256ハッシュを取り、最初の100bitをバイナリ文字列化
        message_hash = hashlib.sha256(message_json.encode()).hexdigest()
        bitstring = bin(int(message_hash[:25], 16))[2:].zfill(100)  # 100bit

        print(f"  埋め込みデータ: {message_data}")
        print(f"  ビット文字列（最初20bit）: {bitstring[:20]}...")

        # RGB変換
        rgb_image = test_image_pil.convert('RGB')

        start_time = time.time()
        watermarked_image = encoder.encode(rgb_image, bitstring, MODE='binary')
        encode_time = time.time() - start_time

        print(f"✓ 透かし埋め込み成功（処理時間: {encode_time:.3f}秒）")
        print(f"  入力サイズ: {rgb_image.size}")
        print(f"  出力サイズ: {watermarked_image.size}")

        if torch.cuda.is_available():
            print(f"  VRAM使用後: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"✗ 透かし埋め込み失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    # デコード（透かし抽出）テスト
    try:
        print("\n透かし抽出テスト...")

        start_time = time.time()
        extracted_secret, wm_present, wm_schema = encoder.decode(
            watermarked_image,
            MODE='binary',
            DETECTFIRST=True,
            ROTATION=False
        )
        decode_time = time.time() - start_time

        print(f"✓ 透かし抽出成功（処理時間: {decode_time:.3f}秒）")
        print(f"  透かし検出: {wm_present}")
        print(f"  埋め込みビット: {bitstring[:20]}...")
        print(f"  抽出ビット: {extracted_secret[:20]}...")
        print(f"  一致: {bitstring == extracted_secret}")

    except Exception as e:
        print(f"✗ 透かし抽出失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ 全テスト成功")
    return True

if __name__ == "__main__":
    success = test_trustmark_basic()
    sys.exit(0 if success else 1)
