#!/usr/bin/env python3
"""
白背景のコントラスト・彩度強調でAI生成ノイズを可視化

使い方:
  python scripts/enhance_noise.py input.png                    # 標準設定
  python scripts/enhance_noise.py input.png -s 5 -c 3          # 彩度5倍、コントラスト3倍
  python scripts/enhance_noise.py input.png -o output.png      # 出力ファイル指定
"""
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

def enhance_noise(image_path: str, saturation: float = 10.0, contrast: float = 5.0, 
                  output_path: str = None) -> str:
    """
    画像の彩度とコントラストを強調してAIノイズを可視化
    
    Args:
        image_path: 入力画像パス
        saturation: 彩度倍率（デフォルト10倍）
        contrast: コントラスト倍率（デフォルト5倍）
        output_path: 出力パス（省略時は _enhanced を付加）
    
    Returns:
        出力ファイルパス
    """
    img = Image.open(image_path).convert("RGB")
    
    # 彩度強調
    enhancer_sat = ImageEnhance.Color(img)
    img = enhancer_sat.enhance(saturation)
    
    # コントラスト強調
    enhancer_con = ImageEnhance.Contrast(img)
    img = enhancer_con.enhance(contrast)
    
    # 出力パス
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_name(f"{p.stem}_enhanced{p.suffix}"))
    
    img.save(output_path)
    return output_path


def create_comparison(image_path: str, saturation: float = 10.0, contrast: float = 5.0,
                      output_path: str = None) -> str:
    """
    オリジナルと強調画像を横並びで比較画像を作成
    """
    img_orig = Image.open(image_path).convert("RGB")
    
    # 強調
    enhancer_sat = ImageEnhance.Color(img_orig)
    img_enhanced = enhancer_sat.enhance(saturation)
    enhancer_con = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer_con.enhance(contrast)
    
    # 横並びに結合
    w, h = img_orig.size
    combined = Image.new("RGB", (w * 2 + 10, h), (128, 128, 128))
    combined.paste(img_orig, (0, 0))
    combined.paste(img_enhanced, (w + 10, 0))
    
    # 出力パス
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_name(f"{p.stem}_comparison{p.suffix}"))
    
    combined.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AI生成ノイズを可視化")
    parser.add_argument("image", help="入力画像パス")
    parser.add_argument("-s", "--saturation", type=float, default=10.0,
                        help="彩度倍率 (default: 10)")
    parser.add_argument("-c", "--contrast", type=float, default=5.0,
                        help="コントラスト倍率 (default: 5)")
    parser.add_argument("-o", "--output", help="出力ファイルパス")
    parser.add_argument("--compare", action="store_true",
                        help="オリジナルと横並び比較画像を作成")
    args = parser.parse_args()
    
    if args.compare:
        out = create_comparison(args.image, args.saturation, args.contrast, args.output)
        print(f"Comparison saved: {out}")
    else:
        out = enhance_noise(args.image, args.saturation, args.contrast, args.output)
        print(f"Enhanced image saved: {out}")


if __name__ == "__main__":
    main()
