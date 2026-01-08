#!/usr/bin/env python3
"""
FastProtect Training Script

CVPR 2025論文「Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models」
の独自実装。

学習対象:
- 3つのMoPセット（y_l, y_m, y_h それぞれに専用の摂動）
  - MoP-L: delta_g_l + Delta_l[k] (y_l用)
  - MoP-M: delta_g_m + Delta_m[k] (y_m用)
  - MoP-H: delta_g_h + Delta_h[k] (y_h用)

Usage:
    # ローカルテスト（小規模）
    python scripts/fastprotect_train.py --test

    # Modal実行
    modal run scripts/fastprotect_train.py --train

    # Modal非同期実行
    modal run scripts/fastprotect_train.py --submit
"""

import modal
from pathlib import Path

app = modal.App("fastprotect-train")

volume = modal.Volume.from_name("fastprotect-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

fastprotect_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",
        "transformers==4.38.2",
        "huggingface_hub==0.21.4",
        "accelerate",
        "safetensors",
        "lpips",
        "pillow",
        "numpy<2.0",
        "tqdm",
        "scikit-learn",
        "kornia",  # DiffJPEG用
    )
)


class FastProtectPerturbations:
    """
    FastProtect摂動管理クラス (3セットMoP対応版)

    論文より:
    - 3つのターゲット (y_l, y_m, y_h) それぞれに専用のMoPを学習
    - 各MoPは delta_g + Delta[k] で構成
    - 各摂動は η/2-ball内に制限（2つ使うので半分ずつ）
    """

    def __init__(
        self,
        K: int = 4,
        image_size: int = 512,
        eta: float = 8 / 255,  # 摂動予算
        device: str = "cuda",
        num_targets: int = 3,  # y_l, y_m, y_h
    ):
        self.K = K
        self.eta = eta
        self.eta_half = eta / 2  # 各摂動の予算は半分
        self.device = device
        self.image_size = image_size
        self.num_targets = num_targets

        # 3セットのMoP
        # MoP[t] = (delta_g_t, [Delta_t_0, ..., Delta_t_K-1])
        self.delta_g = [self._init_perturbation() for _ in range(num_targets)]
        self.Delta = [[self._init_perturbation() for _ in range(K)] for _ in range(num_targets)]

    def _init_perturbation(self):
        """摂動テンソルを初期化"""
        import torch
        import torch.nn as nn

        # 小さなランダム値で初期化
        delta = nn.Parameter(
            torch.randn(3, self.image_size, self.image_size, device=self.device) * 0.001
        )
        return delta

    def get_params(self, target_idx: int = None):
        """
        最適化対象のパラメータを返す

        Args:
            target_idx: 特定のターゲットのパラメータのみ返す（Noneで全て）
        """
        if target_idx is not None:
            return [self.delta_g[target_idx]] + self.Delta[target_idx]

        # 全パラメータ
        params = []
        for t in range(self.num_targets):
            params.append(self.delta_g[t])
            params.extend(self.Delta[t])
        return params

    def apply(self, image, target_idx: int, cluster_idx: int):
        """
        画像に摂動を適用

        Args:
            image: (B, C, H, W) [0, 1]
            target_idx: ターゲットインデックス (0=low, 1=mid, 2=high)
            cluster_idx: クラスタインデックス

        Returns:
            摂動適用済み画像
        """
        import torch

        # 摂動を適用
        delta = self.delta_g[target_idx] + self.Delta[target_idx][cluster_idx]

        # 画像サイズに合わせてリサイズ（必要な場合）
        if image.shape[2:] != (self.image_size, self.image_size):
            import torch.nn.functional as F
            delta_resized = F.interpolate(
                delta.unsqueeze(0),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            delta_resized = delta

        # 適用
        protected = image + delta_resized

        return torch.clamp(protected, 0, 1)

    def clamp_perturbations(self):
        """摂動をη/2-ball内にクランプ"""
        import torch

        with torch.no_grad():
            for t in range(self.num_targets):
                self.delta_g[t].data = torch.clamp(self.delta_g[t].data, -self.eta_half, self.eta_half)
                for delta in self.Delta[t]:
                    delta.data = torch.clamp(delta.data, -self.eta_half, self.eta_half)

    def save(self, path: str, optimizer=None, step: int = None, lambda_: float = None):
        """摂動とOptimizer stateを保存"""
        import torch

        checkpoint = {
            "delta_g": [d.data for d in self.delta_g],
            "Delta": [[d.data for d in deltas] for deltas in self.Delta],
            "K": self.K,
            "eta": self.eta,
            "image_size": self.image_size,
            "num_targets": self.num_targets,
        }

        # Optimizer stateも保存（再開時に必須）
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if step is not None:
            checkpoint["step"] = step
        if lambda_ is not None:
            checkpoint["lambda"] = lambda_

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        """
        摂動をロード

        Returns:
            (instance, checkpoint) - checkpointにはoptimizer_state_dict等が含まれる
        """
        import torch

        checkpoint = torch.load(path, map_location=device)
        instance = cls(
            K=checkpoint["K"],
            image_size=checkpoint["image_size"],
            eta=checkpoint["eta"],
            device=device,
            num_targets=checkpoint.get("num_targets", 3),
        )

        # 3セット分をロード（requires_grad保持のためcopy_を使用）
        for t, d in enumerate(checkpoint["delta_g"]):
            instance.delta_g[t].data.copy_(d.to(device))
        for t, deltas in enumerate(checkpoint["Delta"]):
            for k, d in enumerate(deltas):
                instance.Delta[t][k].data.copy_(d.to(device))

        # requires_gradを明示的に確認・設定（安全策）
        for t in range(instance.num_targets):
            instance.delta_g[t].requires_grad_(True)
            for k in range(instance.K):
                instance.Delta[t][k].requires_grad_(True)

        return instance, checkpoint


def cluster_images_kmeans(latents, K: int = 4):
    """
    K-means++でlatentコードをクラスタリング

    Args:
        latents: (N, C, H, W) のlatentコード
        K: クラスタ数

    Returns:
        cluster_assignments: (N,) のクラスタ割り当て
        kmeans: KMeansモデル（推論時に再利用するため保存必須）
    """
    from sklearn.cluster import KMeans
    import numpy as np

    # (N, C*H*W)にフラット化
    N = latents.shape[0]
    latents_flat = latents.reshape(N, -1)

    if isinstance(latents_flat, np.ndarray) is False:
        latents_flat = latents_flat.cpu().numpy()

    # K-means++
    kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
    cluster_assignments = kmeans.fit_predict(latents_flat)

    return cluster_assignments, kmeans


def predict_cluster(latent, kmeans):
    """
    学習済みK-meansで単一画像のクラスタを予測

    Args:
        latent: (1, C, H, W) または (C, H, W) のlatentコード
        kmeans: 学習済みKMeansモデル

    Returns:
        cluster_idx: クラスタインデックス
    """
    import numpy as np

    if latent.dim() == 4:
        latent = latent.squeeze(0)

    latent_flat = latent.cpu().numpy().reshape(1, -1)
    cluster_idx = kmeans.predict(latent_flat)[0]

    return cluster_idx


def compute_latent_entropy(z):
    """
    Latentコードのエントロピーを計算

    論文のAdaptive Targeted Protectionで使用。
    エントロピーが高い = 複雑なテクスチャ → y_h を使用
    エントロピーが低い = シンプルなテクスチャ → y_l を使用

    Args:
        z: (1, C, H, W) または (B, C, H, W) のlatentコード

    Returns:
        entropy: エントロピー値（スカラーまたは (B,) テンソル）
    """
    import torch

    # 空間方向にフラット化
    if z.dim() == 4:
        B = z.shape[0]
        z_flat = z.view(B, -1)  # (B, C*H*W)
    else:
        z_flat = z.view(1, -1)

    # ヒストグラムベースのエントロピー（簡易版）
    # 値を0-1に正規化してビン分割
    z_normalized = (z_flat - z_flat.min(dim=1, keepdim=True)[0]) / (
        z_flat.max(dim=1, keepdim=True)[0] - z_flat.min(dim=1, keepdim=True)[0] + 1e-8
    )

    # 簡易エントロピー: 分散を使用（分散が大きい = 複雑）
    entropy = z_flat.var(dim=1)

    return entropy


def select_target_by_entropy(z, target_entropies):
    """
    エントロピーに基づいてターゲットを選択

    Args:
        z: (1, C, H, W) の入力latentコード
        target_entropies: [H(y_l), H(y_m), H(y_h)] のリスト

    Returns:
        target_idx: 0=low, 1=mid, 2=high
    """
    import torch

    input_entropy = compute_latent_entropy(z).item()

    # 最も近いターゲットを選択
    distances = [abs(input_entropy - te) for te in target_entropies]
    target_idx = distances.index(min(distances))

    return target_idx


class DifferentiableAugmentation:
    """
    微分可能なデータ拡張（Twitter/圧縮対策）

    学習時にランダムな変換を適用することで、摂動の汎化性能を向上させる。
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def apply(self, x, p: float = 0.5):
        """
        確率pでランダムな拡張を適用

        Args:
            x: (B, C, H, W) [0, 1]
            p: 拡張適用確率

        Returns:
            拡張済み画像
        """
        import torch
        import random

        if random.random() > p:
            return x

        B, C, H, W = x.shape

        # ランダムに変換を選択
        aug_type = random.choice(["resize", "jpeg", "crop"])

        if aug_type == "resize":
            # 微小リサイズ: 480-544 → 512
            scale = random.uniform(0.9375, 1.0625)  # 480/512 to 544/512
            new_size = int(H * scale)
            x = torch.nn.functional.interpolate(x, size=(new_size, new_size), mode="bilinear", align_corners=False)
            x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        elif aug_type == "jpeg":
            # JPEG圧縮シミュレーション（微分可能な近似）
            # 高周波成分を減衰させる
            quality = random.randint(60, 90)
            sigma = (100 - quality) / 20.0  # quality 60 → sigma 2.0, quality 90 → sigma 0.5
            if sigma > 0.1:
                kernel_size = int(sigma * 4) | 1  # 奇数に
                kernel_size = max(3, min(kernel_size, 7))
                x = self._gaussian_blur(x, kernel_size, sigma)

        elif aug_type == "crop":
            # 微小クロップ: 端を数ピクセル切る
            crop_pixels = random.randint(4, 16)
            x = x[:, :, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
            x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return torch.clamp(x, 0, 1)

    def _gaussian_blur(self, x, kernel_size, sigma):
        """ガウシアンブラー"""
        import torch
        import math

        # カーネル作成
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()

        # 分離可能フィルタ
        kernel_h = g.view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
        kernel_v = g.view(1, 1, -1, 1).expand(x.shape[1], 1, -1, 1)

        pad = kernel_size // 2
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = torch.nn.functional.conv2d(x, kernel_h, groups=x.shape[1])
        x = torch.nn.functional.conv2d(x, kernel_v, groups=x.shape[1])

        return x


def enhance_contrast(tensor, strength: float = 1.5):
    """
    テンソル画像のコントラストを強調（ヒストグラム正規化的な効果）

    Args:
        tensor: (1, 3, H, W) [0, 1]
        strength: 強調係数（1.5でギラギラ感、2.0でバキバキ）

    Returns:
        コントラスト強調済みテンソル
    """
    import torch

    # チャンネルごとに正規化してからコントラスト強調
    mean = tensor.mean(dim=(2, 3), keepdim=True)
    enhanced = (tensor - mean) * strength + mean
    return torch.clamp(enhanced, 0, 1)


def get_target_images(size: int = 512, device: str = "cuda", target_dir: str = None, enhance: bool = False):
    """
    ターゲット画像を生成/ロード

    論文のAdaptive Targeted Protectionで使用。
    実写テクスチャが最も効果的（論文準拠）:
    - y_l: レンガ/タイル/畳（規則的な人工物）→ セル塗りキャラに効く
    - y_m: 布地など中間パターン
    - y_h: 森/砂利/芝生（ランダムで細かい自然物）→ 厚塗り背景に効く

    Args:
        size: 画像サイズ
        device: デバイス
        target_dir: ターゲット画像ディレクトリ（Noneで手続き生成）
        enhance: コントラスト強調を適用（VAE消失対策）

    Returns:
        list of 3 target images [y_l, y_m, y_h]
    """
    import torch
    from pathlib import Path
    from PIL import Image
    import torchvision.transforms as T

    targets = []

    # 外部画像があればロード
    if target_dir and Path(target_dir).exists():
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])
        target_files = ["y_l.jpg", "y_m.jpg", "y_h.jpg"]
        for fname in target_files:
            fpath = Path(target_dir) / fname
            if fpath.exists():
                img = Image.open(fpath).convert("RGB")
                t = transform(img).unsqueeze(0).to(device)
                if enhance:
                    t = enhance_contrast(t, strength=1.5)
                targets.append(t)

        if len(targets) == 3:
            print(f"Loaded target images from {target_dir} (contrast enhanced: {enhance})")
            return targets

    # 手続き生成（外部画像がない場合）
    print("Generating procedural target textures...")

    # y_l: レンガ/タイルパターン（規則的な人工物）
    y_l = torch.zeros(1, 3, size, size, device=device)
    brick_h, brick_w = 32, 64
    mortar = 4
    colors = [
        [0.6, 0.3, 0.2],  # 赤茶色レンガ
        [0.55, 0.28, 0.18],
        [0.65, 0.32, 0.22],
    ]
    for i in range(0, size, brick_h + mortar):
        offset = (i // (brick_h + mortar)) % 2 * (brick_w // 2)
        for j in range(-brick_w, size + brick_w, brick_w + mortar):
            jj = j + offset
            if 0 <= jj < size:
                color_idx = (i + j) % 3
                c = colors[color_idx]
                i_end = min(i + brick_h, size)
                j_end = min(jj + brick_w, size)
                jj_start = max(0, jj)
                for ch in range(3):
                    y_l[0, ch, i:i_end, jj_start:j_end] = c[ch]
    # モルタル色（残りの部分）
    mask = y_l.sum(dim=1, keepdim=True) == 0
    y_l[:, 0:1, :, :][mask] = 0.7
    y_l[:, 1:2, :, :][mask.expand(-1, 1, -1, -1)[:, :1]] = 0.7
    y_l[:, 2:3, :, :][mask.expand(-1, 1, -1, -1)[:, :1]] = 0.7
    targets.append(y_l)

    # y_m: 布地パターン（中間）
    y_m = torch.zeros(1, 3, size, size, device=device)
    weave_size = 8
    for i in range(size):
        for j in range(size):
            warp = (i // weave_size) % 2
            weft = (j // weave_size) % 2
            if (i + j) % (weave_size * 2) < weave_size:
                val = 0.4 + 0.2 * warp
            else:
                val = 0.5 + 0.2 * weft
            y_m[0, :, i, j] = val
    torch.manual_seed(42)
    y_m += torch.randn_like(y_m) * 0.05
    y_m = torch.clamp(y_m, 0, 1)
    targets.append(y_m)

    # y_h: 森/芝生パターン（フラクタルノイズで自然テクスチャを模倣）
    torch.manual_seed(123)
    y_h = torch.zeros(1, 3, size, size, device=device)
    for scale in [4, 8, 16, 32, 64, 128]:
        noise = torch.rand(1, 3, size // scale, size // scale, device=device)
        noise_upsampled = torch.nn.functional.interpolate(
            noise, size=(size, size), mode="bilinear", align_corners=False
        )
        y_h += noise_upsampled / (scale ** 0.5)
    # 緑系の色味に調整（森/芝生）
    y_h[:, 0, :, :] *= 0.3
    y_h[:, 1, :, :] *= 0.7
    y_h[:, 2, :, :] *= 0.2
    y_h = (y_h - y_h.min()) / (y_h.max() - y_h.min() + 1e-8)
    targets.append(y_h)

    # コントラスト強調を適用（VAE消失対策）
    if enhance:
        targets = [enhance_contrast(t, strength=1.5) for t in targets]
        print("Applied contrast enhancement to procedural targets")

    return targets


@app.function(
    image=fastprotect_image,
    volumes={VOLUME_PATH: volume},
    gpu="A100",  # A10GだとOOM
    timeout=36000,  # 10時間
)
def train_fastprotect(
    data_dir: str = "/vol/train_images",
    output_dir: str = "/vol/fastprotect_model",
    target_dir: str = None,  # 毒入りターゲット画像ディレクトリ
    num_steps: int = 40000,
    batch_size: int = 8,  # A100使用
    grad_accum_steps: int = 4,  # 
    lr: float = 0.0002,
    eta: float = 8 / 255,
    lambda_: float = None,  # 自動計算（キャリブレーションで決定）
    feature_weight: float = 0.5,  # Feature Lossの相対重み（0.5 = Latent:Feature = 1:0.5）
    warmup_steps: int = 1000,  # Warm-upステップ数
    K: int = 4,
    image_size: int = 512,
    checkpoint_every: int = 5000,
    resume_from: str = None,
):
    """
    FastProtect摂動の学習

    Args:
        data_dir: 学習画像ディレクトリ
        output_dir: 出力ディレクトリ
        num_steps: 学習ステップ数
        batch_size: バッチサイズ
        lr: 学習率
        eta: 摂動予算
        lambda_: MLP Loss中間層重み
        K: クラスタ数
        image_size: 画像サイズ
        checkpoint_every: チェックポイント保存間隔
        resume_from: 再開するチェックポイント
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from diffusers import AutoencoderKL
    from PIL import Image
    from tqdm import tqdm
    import os
    import sys

    # libをパスに追加
    sys.path.insert(0, "/vol/aicheckers")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # VAEロード (bf16で安定性+VRAM節約)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()
    # VAEのrequires_gradは外す（勾配を流すため）
    # ただしoptimizerにはVAEを含めないので重みは更新されない
    # vae.requires_grad_(False)  # 削除: 中間層からの勾配を流す必要あり

    # VAEフック（lib/vae_hooks.pyから）
    # Modalではlib/が使えないのでインラインで定義
    class VAEFeatureExtractor:
        def __init__(self, vae):
            self.features = {}
            self.hooks = []
            self._register_hooks(vae)

        def _make_hook(self, name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.features[name] = output[0]
                else:
                    self.features[name] = output
            return hook

        def _register_hooks(self, vae):
            encoder = vae.encoder
            layers = {
                "down_1": encoder.down_blocks[0],
                "down_2": encoder.down_blocks[1],
                "down_3": encoder.down_blocks[2],
                "mid_0": encoder.mid_block,
            }
            for name, module in layers.items():
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

        def get_feature_list(self):
            return [self.features[n] for n in ["down_1", "down_2", "down_3", "mid_0"] if n in self.features]

        def clear(self):
            self.features = {}

    extractor = VAEFeatureExtractor(vae)

    # データセット
    class ImageDataset(Dataset):
        def __init__(self, root_dir, size):
            self.size = size
            self.files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                self.files.extend(Path(root_dir).glob(ext))
            print(f"Found {len(self.files)} images")
            self.cluster_assignments = None  # 後から設定

        def set_cluster_assignments(self, assignments):
            """K-meansクラスタ割り当てを設定"""
            self.cluster_assignments = assignments

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert("RGB")
            img = img.resize((self.size, self.size), Image.LANCZOS)
            import torchvision.transforms as T
            transform = T.ToTensor()
            img_tensor = transform(img)

            # クラスタ割り当てがあれば返す
            if self.cluster_assignments is not None:
                cluster = self.cluster_assignments[idx]
                return img_tensor, cluster
            return img_tensor, -1  # 未割り当て

    dataset = ImageDataset(data_dir, image_size)
    if len(dataset) == 0:
        print(f"Error: No images found in {data_dir}")
        return {"status": "error", "message": "No images found"}

    # Latentキャッシュのパス
    cache_dir = os.path.join(output_dir, "latent_cache")
    os.makedirs(cache_dir, exist_ok=True)
    latent_cache_path = os.path.join(cache_dir, f"latents_{len(dataset)}_{image_size}.pt")
    cluster_cache_path = os.path.join(cache_dir, f"clusters_{len(dataset)}_{K}.pt")

    # キャッシュがあればロード、なければ計算
    if os.path.exists(latent_cache_path) and os.path.exists(cluster_cache_path):
        print(f"Loading cached latents from {latent_cache_path}...")
        cache_data = torch.load(latent_cache_path)
        all_latents = cache_data["latents"]
        cluster_data = torch.load(cluster_cache_path)
        cluster_assignments = cluster_data["assignments"]
        kmeans_model = cluster_data["kmeans"]
        print(f"Loaded {len(all_latents)} cached latents and cluster assignments")
        print(f"Cluster distribution: {[(cluster_assignments == k).sum() for k in range(K)]}")
    else:
        # 全画像のlatentを事前計算してクラスタリング
        print("Computing latents for clustering (will be cached)...")
        all_latents = []
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Encoding"):
                img_data = dataset[i]
                # タプルの場合は最初の要素（画像）を取得
                img = img_data[0] if isinstance(img_data, tuple) else img_data
                img = img.unsqueeze(0).to(device)
                img_normalized = (img * 2 - 1).bfloat16()
                z = vae.encode(img_normalized).latent_dist.mean.float()
                all_latents.append(z.cpu())

        all_latents = torch.cat(all_latents, dim=0)
        print(f"Latents shape: {all_latents.shape}")

        # K-meansクラスタリング
        print(f"Clustering into {K} groups...")
        cluster_assignments, kmeans_model = cluster_images_kmeans(all_latents, K=K)
        print(f"Cluster distribution: {[(cluster_assignments == k).sum() for k in range(K)]}")

        # キャッシュに保存
        print(f"Saving latent cache to {latent_cache_path}...")
        torch.save({"latents": all_latents}, latent_cache_path)
        import pickle
        torch.save({"assignments": cluster_assignments, "kmeans": kmeans_model}, cluster_cache_path)
        print("Latent cache saved!")

    # データセットにクラスタ割り当てを設定
    dataset.set_cluster_assignments(cluster_assignments)

    # ターゲット画像（毒入りアンカーがあれば使用）
    if target_dir:
        print(f"Loading poisoned target images from {target_dir}...")
    else:
        print("Generating procedural target images...")
    target_images = get_target_images(size=image_size, device=device, target_dir=target_dir)
    num_targets = len(target_images)

    # ターゲットlatentを事前計算 + エントロピー計算
    target_latents = []
    target_features_list = []
    target_entropies = []
    with torch.no_grad():
        for i, t_img in enumerate(target_images):
            t_normalized = (t_img * 2 - 1).bfloat16()
            extractor.clear()
            t_z = vae.encode(t_normalized).latent_dist.mean.float()
            t_features = [f.float() for f in extractor.get_feature_list()]
            target_latents.append(t_z)
            target_features_list.append([f.clone() for f in t_features])
            # エントロピー計算
            t_entropy = compute_latent_entropy(t_z).item()
            target_entropies.append(t_entropy)
            print(f"  Target {i} (y_{'lmh'[i]}): entropy = {t_entropy:.4f}")

    # Differentiable Augmentation
    diff_aug = DifferentiableAugmentation(device=device)
    print("Differentiable Augmentation enabled (p=0.5)")

    # 摂動初期化（3セットMoP対応）
    print(f"Initializing {num_targets} sets of perturbations (K={K} each)...")
    start_step = 0
    saved_optimizer_state = None

    if resume_from and os.path.exists(resume_from):
        perturbations, checkpoint = FastProtectPerturbations.load(resume_from, device=device)
        # チェックポイントからステップ数を取得（なければファイル名から）
        if "step" in checkpoint:
            start_step = checkpoint["step"]
        else:
            import re
            match = re.search(r'step(\d+)', resume_from)
            if match:
                start_step = int(match.group(1))

        # Optimizer stateがあれば保存
        if "optimizer_state_dict" in checkpoint:
            saved_optimizer_state = checkpoint["optimizer_state_dict"]
            print(f"Found optimizer state in checkpoint")
        else:
            print(f"WARNING: No optimizer state in checkpoint - training may be unstable!")

        # λがあれば復元
        if "lambda" in checkpoint:
            lambda_ = checkpoint["lambda"]
            print(f"Restored λ = {lambda_:.2e} from checkpoint")

        print(f"Resumed from {resume_from}, starting at step {start_step}")
    else:
        perturbations = FastProtectPerturbations(
            K=K, image_size=image_size, eta=eta, device=device, num_targets=num_targets
        )

    # Optimizer（全パラメータを一括で最適化）
    optimizer = torch.optim.Adam(
        perturbations.get_params(),
        lr=lr,
        betas=(0.5, 0.99),
    )

    # Optimizer stateを復元
    if saved_optimizer_state is not None:
        optimizer.load_state_dict(saved_optimizer_state)
        print(f"Restored optimizer state (momentum/velocity preserved)")

    # Warm-up学習率スケジューラ（Linear warm-up）
    def get_lr_scale(step):
        """Warm-up: 0→warmup_stepsで0→1に線形増加、以降は1.0"""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    # 新規学習時のみWarm-upを適用（再開時は既にmotmentumがあるのでスキップ）
    use_warmup = (saved_optimizer_state is None and start_step == 0)
    if use_warmup:
        print(f"Using linear warm-up for first {warmup_steps} steps (lr: 0 → {lr})")
    else:
        print(f"Warm-up skipped (resuming from step {start_step})")

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    dataloader_iter = iter(dataloader)

    # 学習ループ
    # 二次元イラスト特化: シンプルテクスチャ(y_l)を重視した重み付きサンプリング
    # y_l: 45% (セル塗り、白背景、顔アップ)
    # y_m: 35% (服の装飾、髪ハイライト)
    # y_h: 20% (星空、風景背景、厚塗り)
    import random
    target_weights = [0.45, 0.35, 0.20]

    # === λキャリブレーション ===
    # 最初の10バッチでLatent/Feature Lossのスケールを計測し、λを自動決定
    calibration_steps = 10
    if lambda_ is None and start_step == 0:
        print(f"Calibrating lambda over {calibration_steps} batches...")
        latent_losses_cal = []
        feature_losses_cal = []

        cal_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        cal_iter = iter(cal_dataloader)

        with torch.no_grad():
            for cal_step in range(calibration_steps):
                try:
                    cal_batch_data = next(cal_iter)
                except StopIteration:
                    cal_iter = iter(cal_dataloader)
                    cal_batch_data = next(cal_iter)

                cal_batch, cal_clusters = cal_batch_data
                cal_batch = cal_batch.to(device)
                B_cal = cal_batch.shape[0]

                # ターゲット選択
                cal_target_idx = random.choices(range(num_targets), weights=target_weights)[0]
                cal_target_z = target_latents[cal_target_idx]
                cal_target_features = target_features_list[cal_target_idx]

                # VAE encode（摂動なし）
                extractor.clear()
                cal_normalized = (cal_batch * 2 - 1).bfloat16()
                cal_z = vae.encode(cal_normalized).latent_dist.mean.float()
                cal_features = [f.float() for f in extractor.get_feature_list()]

                # Latent Loss（Raw）
                cal_target_z_exp = cal_target_z.expand(B_cal, -1, -1, -1)
                cal_latent_loss = torch.mean((cal_z - cal_target_z_exp) ** 2).item()
                latent_losses_cal.append(cal_latent_loss)

                # Feature Loss（λ=1で計算、Raw）
                cal_target_feat_exp = [f.expand(B_cal, -1, -1, -1) for f in cal_target_features]
                L = len(cal_features)
                cal_feature_loss = 0.0
                for f, f_t in zip(cal_features, cal_target_feat_exp):
                    cal_feature_loss += (1.0 / L) * torch.mean((f - f_t) ** 2).item()
                feature_losses_cal.append(cal_feature_loss)

        avg_latent = sum(latent_losses_cal) / len(latent_losses_cal)
        avg_feature = sum(feature_losses_cal) / len(feature_losses_cal)

        # λ = (latent / feature) * feature_weight
        # これにより feature_loss * λ ≈ latent_loss * feature_weight
        lambda_ = (avg_latent / avg_feature) * feature_weight
        print(f"Calibration complete:")
        print(f"  Avg Latent Loss (raw): {avg_latent:.4f}")
        print(f"  Avg Feature Loss (raw, λ=1): {avg_feature:.4f}")
        print(f"  Auto λ = {lambda_:.2e} (feature_weight={feature_weight})")
        print(f"  Expected ratio: Latent:Feature ≈ 1:{feature_weight}")
    elif lambda_ is None:
        # 再開時はデフォルト値を使用
        lambda_ = 3.5e-5
        print(f"Using default λ={lambda_} for resumed training")
    else:
        print(f"Using provided λ={lambda_}")

    print(f"Starting training for {num_steps} steps (3 MoP sets, weights={target_weights})...")
    losses = {t: [] for t in range(num_targets)}  # ターゲットごとのloss
    import time
    start_time = time.time()
    optimizer.zero_grad()  # 勾配蓄積用に初期化

    for step in tqdm(range(start_step, num_steps), desc="Training"):
        # バッチ取得（DataLoaderが終わったら再初期化）
        try:
            batch_data = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch_data = next(dataloader_iter)

        # (images, clusters) のタプルを展開
        batch, batch_clusters = batch_data
        batch = batch.to(device)
        batch_clusters = batch_clusters.to(device)
        B = batch.shape[0]

        # ターゲット選択: 重み付きサンプリング（二次元イラスト特化）
        target_idx = random.choices(range(num_targets), weights=target_weights)[0]
        target_z = target_latents[target_idx]
        target_features = target_features_list[target_idx]

        # バッチ全体に摂動を適用（クラスタごとに異なる摂動）
        protected_list = []
        for i in range(B):
            k = batch_clusters[i].item()
            img_i = batch[i:i+1]
            protected_i = perturbations.apply(img_i, target_idx, k)
            protected_list.append(protected_i)
        protected = torch.cat(protected_list, dim=0)

        # Differentiable Augmentation
        protected = diff_aug.apply(protected, p=0.5)

        # バッチ全体を一度にVAE encode（効率化）
        extractor.clear()
        protected_normalized = (protected * 2 - 1).bfloat16()
        z = vae.encode(protected_normalized).latent_dist.mean.float()
        features = [f.float() for f in extractor.get_feature_list()]

        # ターゲットをバッチサイズに拡張
        target_z_expanded = target_z.expand(B, -1, -1, -1)
        target_features_expanded = [f.expand(B, -1, -1, -1) for f in target_features]

        # Latent距離（最小化 = ターゲットに近づける）
        latent_loss = torch.mean((z - target_z_expanded) ** 2)

        # 中間層距離（最小化）- 勾配を流す
        L = len(features)
        feature_loss = 0.0
        for f, f_t in zip(features, target_features_expanded):
            feature_loss += (lambda_ / L) * torch.mean((f - f_t) ** 2)

        total_loss = (latent_loss + feature_loss) / grad_accum_steps

        # 勾配蓄積
        total_loss.backward()

        # 勾配デバッグ（最初の数ステップだけ）
        if step < start_step + 5:
            grad_mean = perturbations.delta_g[0].grad
            if grad_mean is not None:
                print(f"DEBUG step {step}: delta.grad mean = {grad_mean.abs().mean().item():.6f}")
            else:
                print(f"DEBUG step {step}: delta.grad is None! 勾配が流れていない!")

        # grad_accum_stepsごとに更新
        if (step + 1) % grad_accum_steps == 0:
            # Warm-up: 学習率をスケール
            if use_warmup:
                lr_scale = get_lr_scale(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale

            optimizer.step()
            optimizer.zero_grad()
            perturbations.clamp_perturbations()

        losses[target_idx].append((latent_loss + feature_loss).item())  # 合計を記録

        # ログ（100ステップごと）- latent/feature比率も表示
        if (step + 1) % 100 == 0:
            avg_losses = {t: sum(losses[t][-100:]) / max(1, len(losses[t][-100:])) for t in range(num_targets)}
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - start_time if 'start_time' in dir() else 0
            img_per_sec = (step + 1) * batch_size / elapsed if elapsed > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step + 1}/{num_steps}, Loss: L={avg_losses[0]:.4f}, M={avg_losses[1]:.4f}, H={avg_losses[2]:.4f}")
            print(f"  latent={latent_loss.item():.4f}, feat={feature_loss:.6f}, lr={current_lr:.2e}, VRAM={vram_gb:.1f}GB, {img_per_sec:.1f}img/s")

        # チェックポイント保存（Optimizer state + λも含む）
        if (step + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step{step + 1}.pt")
            perturbations.save(ckpt_path, optimizer=optimizer, step=step + 1, lambda_=lambda_)
            print(f"Saved checkpoint to {ckpt_path} (with optimizer state, λ={lambda_:.2e})")
            # volumeにも同期（途中で落ちても復旧可能に）
            volume.commit()

    # 最終保存
    final_path = os.path.join(output_dir, "fastprotect_final.pt")
    perturbations.save(final_path, optimizer=optimizer, step=num_steps, lambda_=lambda_)

    # K-meansモデルも保存（推論時に必要）
    import pickle
    kmeans_path = os.path.join(output_dir, "kmeans_model.pkl")
    with open(kmeans_path, "wb") as f:
        pickle.dump(kmeans_model, f)

    # ターゲットエントロピーを保存（推論時のターゲット選択に必要）
    import json
    entropy_path = os.path.join(output_dir, "target_entropies.json")
    with open(entropy_path, "w") as f:
        json.dump({"entropies": target_entropies}, f)

    print(f"Training complete! Saved to {final_path}")
    print(f"K-means model saved to {kmeans_path}")
    print(f"Target entropies saved to {entropy_path}")

    # volumeに同期
    volume.commit()

    return {
        "status": "success",
        "model_path": final_path,
        "final_loss": losses[-1] if losses else None,
        "num_steps": num_steps,
    }


@app.function(
    image=fastprotect_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=600,
)
def test_training_small():
    """小規模テスト（動作確認用）"""
    import torch
    import os
    from PIL import Image
    import numpy as np

    # テスト画像を生成
    test_dir = f"{VOLUME_PATH}/test_train_images"
    os.makedirs(test_dir, exist_ok=True)

    # ダミー画像を作成
    for i in range(32):
        img = Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
        img.save(f"{test_dir}/test_{i:03d}.png")

    # 小規模学習
    result = train_fastprotect.local(
        data_dir=test_dir,
        output_dir=f"{VOLUME_PATH}/test_fastprotect",
        num_steps=100,
        batch_size=4,
        checkpoint_every=50,
    )

    return result


@app.local_entrypoint()
def main(
    train: bool = False,
    test: bool = False,
    submit: bool = False,
    data_dir: str = None,
    target_dir: str = None,
    steps: int = 40000,
    use_poisoned_anchors: bool = False,
    resume_from: str = None,
):
    """
    エントリポイント

    Args:
        train: 学習を実行
        test: 小規模テストを実行
        submit: 非同期ジョブとして投入
        data_dir: 学習データディレクトリ
        target_dir: ターゲット画像ディレクトリ（毒入りアンカー用）
        steps: 学習ステップ数
        use_poisoned_anchors: 毒入りアンカーを使用（target_dirを自動設定）
    """
    if test:
        print("Running small scale test...")
        result = test_training_small.remote()
        print(f"Result: {result}")

    elif train or submit:
        if data_dir is None:
            data_dir = f"{VOLUME_PATH}/train_images"

        # 毒入りアンカーを使用する場合
        if use_poisoned_anchors and target_dir is None:
            target_dir = f"{VOLUME_PATH}/fastprotect_targets"
            print(f"Using poisoned anchors from {target_dir}")

        print(f"Starting training with data from {data_dir}")
        if target_dir:
            print(f"Target images from: {target_dir}")
        print(f"Steps: {steps}")

        if resume_from:
            print(f"Resuming from: {resume_from}")

        if submit:
            # 非同期投入
            call = train_fastprotect.spawn(
                data_dir=data_dir,
                target_dir=target_dir,
                num_steps=steps,
                resume_from=resume_from,
            )
            print(f"Job submitted! Call ID: {call.object_id}")
        else:
            # 同期実行
            result = train_fastprotect.remote(
                data_dir=data_dir,
                target_dir=target_dir,
                num_steps=steps,
                resume_from=resume_from,
            )
            print(f"Result: {result}")
    else:
        print("Usage:")
        print("  modal run scripts/fastprotect_train.py --test")
        print("  modal run scripts/fastprotect_train.py --train --data-dir /vol/images")
        print("  modal run scripts/fastprotect_train.py --submit --data-dir /vol/images")
        print("")
        print("With poisoned anchors (Gemini strategy):")
        print("  modal run scripts/fastprotect_train.py --train --use-poisoned-anchors")
        print("  modal run scripts/fastprotect_train.py --submit --use-poisoned-anchors")


if __name__ == "__main__":
    # ローカルテスト用
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print("Local test mode - use 'modal run' for actual execution")
