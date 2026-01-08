#!/usr/bin/env python3
"""
TrustMark透かしがFastProtect後も生き残っているかテスト
Original → TrustMark埋め込み → FastProtect → 抽出テスト
"""
import sys
sys.path.insert(0, '/home/techne/aicheckers')

from pathlib import Path
from PIL import Image
import trustmark

ORIGINAL_DIR = Path("target_images")
PROTECTED_DIR = Path("temp_protected_output")

def main():
    print("=" * 60)
    print("TrustMark Survival Test: After FastProtect")
    print("=" * 60)
    
    # TrustMark初期化
    print("\nInitializing TrustMark decoder...")
    decoder = trustmark.TrustMark()
    print(f"Schema capacity: {decoder.schemaCapacity()} bits")
    
    results = []
    
    for protected_path in sorted(PROTECTED_DIR.glob("*")):
        if protected_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        
        print(f"\n--- {protected_path.name} ---")
        
        try:
            # 保護済み画像を読み込み
            prot_img = Image.open(protected_path).convert("RGB")
            print(f"Size: {prot_img.size}")
            
            # 透かし抽出を試みる
            extracted_secret, wm_present, wm_schema = decoder.decode(
                prot_img,
                MODE='binary',
                DETECTFIRST=True,
                ROTATION=False
            )
            
            if wm_present:
                print(f"✓ WATERMARK DETECTED!")
                print(f"  Extracted bits (first 20): {extracted_secret[:20]}...")
                results.append((protected_path.name, True, extracted_secret))
            else:
                print(f"✗ No watermark detected")
                results.append((protected_path.name, False, ""))
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((protected_path.name, False, str(e)))
    
    # サマリー
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    detected = sum(1 for _, found, _ in results if found)
    total = len(results)
    
    print(f"Detected: {detected}/{total}")
    
    if detected == total:
        print("\n✓ CONCLUSION: TrustMark watermarks SURVIVE FastProtect!")
        print("  → Hybrid TrustMark + DINOv3 tracking is VIABLE.")
    elif detected > 0:
        print(f"\n⚠ CONCLUSION: Partial survival ({detected}/{total})")
        print("  → Some images lose watermarks. Investigation needed.")
    else:
        print("\n✗ CONCLUSION: TrustMark watermarks DO NOT survive FastProtect.")
        print("  → Need to adjust embedding strength or order of operations.")

if __name__ == "__main__":
    main()
