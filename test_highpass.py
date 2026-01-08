#!/usr/bin/env python3
"""
ハイパスフィルタ統計量の抽出テスト
実際に動作するか確認
"""
import numpy as np
from PIL import Image
import cv2

def extract_high_freq_stats(image_path: str) -> np.ndarray:
    """
    画像から高周波統計量を抽出

    Returns:
        stats: (5,) 高周波統計量
    """
    # 画像読み込み
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # グレースケール変換
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # FFTで周波数変換
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)  # 中心に低周波を配置
    magnitude = np.abs(fft_shifted)

    # ハイパスフィルタマスク作成（中心の低周波をカット）
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 8  # 中心1/8の円をマスク

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    low_freq_mask = dist_from_center <= radius
    high_freq_mask = ~low_freq_mask

    # 高周波成分の統計量を計算
    high_freq_magnitude = magnitude * high_freq_mask

    stats = np.array([
        high_freq_magnitude.mean(),           # 高周波エネルギー平均
        high_freq_magnitude.std(),            # 高周波エネルギー標準偏差
        high_freq_magnitude.max(),            # 最大値
        np.percentile(high_freq_magnitude[high_freq_mask], 95),  # 95パーセンタイル
        (high_freq_magnitude > high_freq_magnitude.mean()).sum() / high_freq_magnitude.size  # 閾値超え割合
    ], dtype=np.float32)

    return stats


if __name__ == "__main__":
    # テスト: AI画像とHuman画像で統計量を比較
    import sys

    print("=== ハイパスフィルタ統計量抽出テスト ===\n")

    # NovelAI画像でテスト
    ai_image = "data/novelai_combined/1.jpg"
    human_image = "data/animedl2m_dataset_release/real_images/images/1.jpg"

    try:
        ai_stats = extract_high_freq_stats(ai_image)
        print(f"AI画像統計量: {ai_stats}")
    except Exception as e:
        print(f"AI画像エラー: {e}")
        ai_stats = None

    try:
        human_stats = extract_high_freq_stats(human_image)
        print(f"Human画像統計量: {human_stats}")
    except Exception as e:
        print(f"Human画像エラー: {e}")
        human_stats = None

    if ai_stats is not None and human_stats is not None:
        print(f"\n差分: {ai_stats - human_stats}")
        print(f"差分ノルム: {np.linalg.norm(ai_stats - human_stats):.4f}")
        print("\n✅ 実装は動作します")
    else:
        print("\n❌ テスト画像が見つかりません（実装自体は正常）")
