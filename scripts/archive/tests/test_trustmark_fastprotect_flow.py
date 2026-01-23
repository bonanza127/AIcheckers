#!/usr/bin/env python3
"""
完全なフローテスト: Original → TrustMark埋め込み → MoonKnight(FastProtect) → 透かし抽出
"""
import sys
sys.path.insert(0, '/home/techne/aicheckers')

from pathlib import Path
from PIL import Image
import trustmark
import numpy as np
import torch

# MoonKnight V3のインポート
from scripts.moonknight_v3 import MoonKnightV3

ORIGINAL_DIR = Path("target_images")
OUTPUT_DIR = Path("temp_trustmark_fastprotect_test")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 70)
    print("Full Flow Test: TrustMark → MoonKnight → Extraction")
    print("=" * 70)
    
    # TrustMark初期化
    print("\nInitializing TrustMark...")
    tm = trustmark.TrustMark()
    capacity = tm.schemaCapacity()
    print(f"Schema capacity: {capacity} bits")
    
    # MoonKnight初期化
    print("\nInitializing MoonKnight V3...")
    moonknight = MoonKnightV3(
        model_dir="/home/techne/aicheckers/models/fastprotect",
        device="cuda",
        use_adaptive=True
    )
    print("MoonKnight V3 ready")
    
    # テスト用透かしデータ
    test_bitstring = "1" * capacity  # 全て1のパターン（検証しやすい）
    print(f"Test watermark: {'1' * 20}... ({capacity} bits)")
    
    results = []
    
    for orig_path in sorted(ORIGINAL_DIR.glob("*")):
        if orig_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {orig_path.name}")
        print("=" * 70)
        
        try:
            # 1. オリジナル画像読み込み
            orig_img = Image.open(orig_path).convert("RGB")
            print(f"1. Original size: {orig_img.size}")
            
            # サイズが小さすぎる場合はスキップ
            if orig_img.size[0] < 256 or orig_img.size[1] < 256:
                print(f"   ⚠ Image too small for reliable watermarking, skipping")
                results.append((orig_path.name, "SKIP", "too small", 0))
                continue
            
            # 2. TrustMark透かし埋め込み
            print("2. Embedding TrustMark watermark...")
            watermarked_img = tm.encode(orig_img, test_bitstring, MODE='binary')
            print(f"   Watermarked size: {watermarked_img.size}")
            
            # 埋め込み確認
            _, wm_check, _ = tm.decode(watermarked_img, MODE='binary', DETECTFIRST=True, ROTATION=False)
            print(f"   Watermark embedded: {'✓' if wm_check else '✗'}")
            
            # 3. MoonKnight適用（PIL Image → PIL Image）
            print("3. Applying MoonKnight protection...")
            protected_img = moonknight.poison(watermarked_img, strength=0.6)
            print(f"   Protected size: {protected_img.size}")
            
            # 保存
            out_path = OUTPUT_DIR / orig_path.name
            protected_img.save(out_path, quality=95)
            
            # 4. 透かし抽出テスト
            print("4. Extracting watermark from protected image...")
            extracted_secret, wm_present, _ = tm.decode(
                protected_img, 
                MODE='binary', 
                DETECTFIRST=True, 
                ROTATION=False
            )
            
            if wm_present:
                # ビット一致率計算
                match_count = sum(1 for a, b in zip(test_bitstring, extracted_secret) if a == b)
                accuracy = match_count / len(test_bitstring) * 100
                print(f"   ✓ WATERMARK DETECTED!")
                print(f"   Extracted: {extracted_secret[:20]}...")
                print(f"   Bit accuracy: {accuracy:.1f}%")
                results.append((orig_path.name, "DETECTED", extracted_secret, accuracy))
            else:
                print(f"   ✗ No watermark detected")
                results.append((orig_path.name, "NOT_DETECTED", "", 0))
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((orig_path.name, "ERROR", str(e), 0))
    
    # サマリー
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    detected = sum(1 for _, status, _, _ in results if status == "DETECTED")
    not_detected = sum(1 for _, status, _, _ in results if status == "NOT_DETECTED")
    skipped = sum(1 for _, status, _, _ in results if status == "SKIP")
    errors = sum(1 for _, status, _, _ in results if status == "ERROR")
    
    print(f"Detected:     {detected}")
    print(f"Not Detected: {not_detected}")
    print(f"Skipped:      {skipped}")
    print(f"Errors:       {errors}")
    
    if detected > 0:
        avg_accuracy = sum(acc for _, status, _, acc in results if status == "DETECTED") / detected
        print(f"Avg Bit Accuracy: {avg_accuracy:.1f}%")
    
    if detected == len(results) - skipped - errors:
        print("\n✓ CONCLUSION: TrustMark watermarks SURVIVE FastProtect!")
    elif detected > 0:
        print(f"\n⚠ CONCLUSION: Partial survival ({detected}/{len(results) - skipped - errors})")
    else:
        print("\n✗ CONCLUSION: TrustMark watermarks DO NOT survive FastProtect.")

if __name__ == "__main__":
    main()
