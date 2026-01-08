#!/usr/bin/env python3
"""
VAE Feature Extractor with Hooks

FastProtect論文のMulti-Layer Protection (MLP) Lossに必要な中間層特徴量を取得。

対象層（論文より）:
- down_1: encoder.down_blocks[0]
- down_2: encoder.down_blocks[1]
- down_3: encoder.down_blocks[2]
- mid_0: encoder.mid_block
"""

import torch
from typing import Dict, List, Tuple, Optional


class VAEFeatureExtractor:
    """
    VAEエンコーダの中間層特徴量を取得するためのフック管理クラス

    Usage:
        vae = AutoencoderKL.from_pretrained(...)
        extractor = VAEFeatureExtractor(vae)

        # 順伝播
        latent = vae.encode(image).latent_dist.mean
        features = extractor.get_features()
        extractor.clear()
    """

    def __init__(self, vae, layer_names: Optional[List[str]] = None):
        """
        Args:
            vae: diffusers.AutoencoderKL インスタンス
            layer_names: 取得する層の名前リスト（Noneで全層）
        """
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_names = layer_names or ["down_1", "down_2", "down_3", "mid_0"]

        self._register_hooks(vae)

    def _make_hook(self, name: str):
        """指定名でフックを生成"""
        def hook(module, input, output):
            # 出力が複数の場合は最初の要素（hidden_states）を取得
            if isinstance(output, tuple):
                self.features[name] = output[0]
            else:
                self.features[name] = output
        return hook

    def _register_hooks(self, vae):
        """VAEエンコーダの指定層にフックを登録"""
        encoder = vae.encoder

        # 層名とモジュールのマッピング
        layer_mapping = {
            "down_1": encoder.down_blocks[0],
            "down_2": encoder.down_blocks[1],
            "down_3": encoder.down_blocks[2],
            "mid_0": encoder.mid_block,
        }

        for name in self.layer_names:
            if name in layer_mapping:
                module = layer_mapping[name]
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
            else:
                print(f"Warning: Layer '{name}' not found in VAE encoder")

    def get_features(self) -> Dict[str, torch.Tensor]:
        """取得した特徴量を返す"""
        return self.features

    def get_feature_list(self) -> List[torch.Tensor]:
        """特徴量をリストとして返す（順序保証）"""
        return [self.features[name] for name in self.layer_names if name in self.features]

    def clear(self):
        """特徴量バッファをクリア"""
        self.features = {}

    def remove_hooks(self):
        """全フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """デストラクタでフックを削除"""
        self.remove_hooks()


def verify_vae_hooks(vae, device: str = "cuda") -> Dict[str, Tuple[int, ...]]:
    """
    VAEフックの動作検証

    Args:
        vae: diffusers.AutoencoderKL インスタンス
        device: デバイス

    Returns:
        各層の出力形状の辞書
    """
    extractor = VAEFeatureExtractor(vae)

    # ダミー入力で順伝播
    dummy_input = torch.randn(1, 3, 512, 512, device=device)
    _ = vae.encode(dummy_input).latent_dist.mean

    # 形状を確認
    shapes = {}
    for name, feature in extractor.get_features().items():
        shapes[name] = tuple(feature.shape)

    extractor.remove_hooks()
    return shapes


if __name__ == "__main__":
    # テスト用
    from diffusers import AutoencoderKL

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float16,
    ).to("cuda")

    print("Verifying hooks...")
    shapes = verify_vae_hooks(vae, device="cuda")
    for name, shape in shapes.items():
        print(f"  {name}: {shape}")
