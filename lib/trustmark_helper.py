"""
TrustMark透かし埋め込み・抽出のヘルパー関数
"""
import json
import hashlib
from typing import Dict, Optional, Tuple
from PIL import Image


def encode_user_data_to_bitstring(user_id: str, timestamp: str, capacity: int = 61) -> str:
    """
    user_idとタイムスタンプを61bitのバイナリ文字列に変換

    Args:
        user_id: ユーザーID
        timestamp: ISO形式のタイムスタンプ
        capacity: 透かし容量（bit）

    Returns:
        バイナリ文字列（'0'と'1'のみ）
    """
    data = {"user_id": user_id, "timestamp": timestamp}
    data_json = json.dumps(data, separators=(',', ':'))  # 圧縮

    # SHA256ハッシュ化
    hash_hex = hashlib.sha256(data_json.encode()).hexdigest()

    # 最初のN bit分をバイナリに変換
    bits_needed = (capacity + 3) // 4  # 4bit/hex文字
    binary = bin(int(hash_hex[:bits_needed], 16))[2:]  # '0b'プレフィックス除去

    # 指定容量にゼロパディング
    bitstring = binary.zfill(capacity)[:capacity]

    return bitstring


def embed_watermark(
    trustmark_encoder,
    image: Image.Image,
    user_id: str,
    timestamp: str,
    alpha: float = 1.15
) -> Image.Image:
    """
    画像にTrustMark透かしを埋め込む

    Args:
        trustmark_encoder: TrustMarkエンコーダーインスタンス
        image: RGB画像
        user_id: ユーザーID
        timestamp: ISO形式のタイムスタンプ
        alpha: 透かし強度（デフォルト1.15、FastProtect併用時の推奨値）

    Returns:
        透かし入り画像
    """
    # 容量確認
    capacity = trustmark_encoder.schemaCapacity()

    # user_id + timestampをbitstringに変換
    bitstring = encode_user_data_to_bitstring(user_id, timestamp, capacity)

    # RGB変換（すでにRGBの場合は変わらない）
    rgb_image = image.convert('RGB')

    # 透かし埋め込み（alpha引数はTrustMark内部で使用される想定）
    # 注: 公式APIにalphaパラメータがない場合、カスタム実装が必要
    watermarked_image = trustmark_encoder.encode(rgb_image, bitstring, MODE='binary')

    return watermarked_image


def extract_watermark(
    trustmark_decoder,
    image: Image.Image
) -> Tuple[bool, str, str]:
    """
    画像からTrustMark透かしを抽出

    Args:
        trustmark_decoder: TrustMarkデコーダーインスタンス
        image: 透かし入り画像

    Returns:
        (is_detected, user_id, timestamp)
        - is_detected: 透かしが検出されたか
        - user_id: ユーザーID（検出失敗時は空文字列）
        - timestamp: タイムスタンプ（検出失敗時は空文字列）
    """
    try:
        # RGB変換
        rgb_image = image.convert('RGB')

        # 透かし抽出
        extracted_secret, wm_present, wm_schema = trustmark_decoder.decode(
            rgb_image,
            MODE='binary',
            DETECTFIRST=True,
            ROTATION=False
        )

        if not wm_present:
            return False, "", ""

        # bitstringから元データを復元（ハッシュなので直接復元不可）
        # 代わりに、bitstringをIDとして使用
        # 実用的には、user_id -> bitstring のマッピングテーブルを持つ

        # 現時点では、検出成功のみ返す
        # TODO: user_id逆引き機能の実装
        return True, "detected", extracted_secret

    except Exception as e:
        print(f"Watermark extraction error: {e}")
        return False, "", ""


def create_user_watermark_mapping(user_id: str, timestamp: str, capacity: int = 61) -> str:
    """
    user_id + timestampのハッシュを生成（DB保存用）

    Args:
        user_id: ユーザーID
        timestamp: タイムスタンプ
        capacity: 透かし容量

    Returns:
        ハッシュ文字列（検索キーとして使用）
    """
    bitstring = encode_user_data_to_bitstring(user_id, timestamp, capacity)
    return bitstring
