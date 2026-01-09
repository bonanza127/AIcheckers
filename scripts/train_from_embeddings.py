#!/usr/bin/env python3
"""
保存済みembeddingsからLinear Probe分類器を学習
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
OUTPUT_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")

# AIカテゴリ（1=AI）- 全データ
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    "sdxl10_ai",
    "sd15_ai",
    "other_ai",
    "flux1d_ai",
    "novelai_ai",
    "pixai_ai",
    "novelai_aibooru_ai",
    # "novelai_pixiv_ai",  # 品質が低いため除外
    # "twitter_novelai_all_ai",  # 古い（重複多し）
    "novelai_combined_ai",  # dedup済みPixiv+Twitter (4,499枚)
    "pixiv_novelai_v2_ai",  # Pixiv NovelAI dedup済み (8,859枚)
    "twitter_novelai_v2_ai",  # Twitter NovelAI dedup済み (12,262枚)
    "novelai_artist_tagged_ai",  # AIBooru アーティストタグ付き (846枚)
]

# Realカテゴリ（0=Real）
REAL_CATEGORIES = [
    "danbooru_real",
]


def load_embeddings(use_patch_stats=True):
    """全embeddingsを読み込んで結合し、クラスバランスを調整"""
    ai_embeddings = []
    real_embeddings = []

    print("Loading AI embeddings...")
    for cat in AI_CATEGORIES:
        cls_path = EMBEDDINGS_DIR / f"{cat}.npy"
        stats_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"

        if not cls_path.exists():
            print(f"  {cat}: CLS NOT FOUND, skipping")
            continue

        cls_emb = np.load(cls_path)

        # SD1.5制限（10,000枚）
        if cat == "sd15_ai" and cls_emb.shape[0] > 10000:
            print(f"  {cat}: Capping {cls_emb.shape[0]} -> 10000")
            indices = np.random.choice(cls_emb.shape[0], 10000, replace=False)
            cls_emb = cls_emb[indices]

        if use_patch_stats:
            if not stats_path.exists():
                print(f"  {cat}: patch_stats NOT FOUND, skipping")
                continue
            stats = np.load(stats_path)
            if cat == "sd15_ai" and stats.shape[0] > 10000:
                stats = stats[indices] # 使用したインデックスを再利用
            
            # CLS + patch_stats を結合
            emb = np.concatenate([cls_emb, stats], axis=1)
            print(f"  {cat}: {emb.shape[0]} samples ({emb.shape[1]} dims)")
        else:
            emb = cls_emb
            print(f"  {cat}: {emb.shape[0]} samples (768 dims)")

        ai_embeddings.append(emb)

    print("\nLoading Real embeddings...")
    for cat in REAL_CATEGORIES:
        cls_path = EMBEDDINGS_DIR / f"{cat}.npy"
        stats_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"

        if not cls_path.exists():
            print(f"  {cat}: CLS NOT FOUND, skipping")
            continue

        cls_emb = np.load(cls_path)

        if use_patch_stats:
            if not stats_path.exists():
                print(f"  {cat}: patch_stats NOT FOUND, skipping")
                continue
            stats = np.load(stats_path)
            emb = np.concatenate([cls_emb, stats], axis=1)
            print(f"  {cat}: {emb.shape[0]} samples ({emb.shape[1]} dims)")
        else:
            emb = cls_emb
            print(f"  {cat}: {emb.shape[0]} samples (768 dims)")

        real_embeddings.append(emb)

    if not ai_embeddings or not real_embeddings:
        raise ValueError("No embeddings loaded!")

    # 一時結合
    ai_all = np.concatenate(ai_embeddings, axis=0)
    real_all = np.concatenate(real_embeddings, axis=0)

    n_ai = ai_all.shape[0]
    n_real = real_all.shape[0]
    
    print(f"\nBefore balancing: AI={n_ai}, Real={n_real}")
    
    # バランス調整（少ない方に合わせる）
    n_target = min(n_ai, n_real)
    
    if n_ai > n_target:
        print(f"Subsampling AI: {n_ai} -> {n_target}")
        idx = np.random.choice(n_ai, n_target, replace=False)
        ai_all = ai_all[idx]
        
    if n_real > n_target:
        print(f"Subsampling Real: {n_real} -> {n_target}")
        idx = np.random.choice(n_real, n_target, replace=False)
        real_all = real_all[idx]

    print(f"Final Count: AI={ai_all.shape[0]}, Real={real_all.shape[0]} (Total {ai_all.shape[0]*2})")
    print(f"Feature dimension: {ai_all.shape[1]}")

    # ラベル作成
    ai_labels = np.ones(ai_all.shape[0])
    real_labels = np.zeros(real_all.shape[0])

    X = np.concatenate([ai_all, real_all], axis=0)
    y = np.concatenate([ai_labels, real_labels], axis=0)

    return X, y, ai_all.shape[1]  # input_dimも返す


def train_classifier(X, y, input_dim, epochs=60, lr=0.001,
                     vat_epsilon=0.005, vat_alpha_start=0.05, vat_alpha_end=0.3,
                     entropy_start_epoch=20, entropy_alpha_end=0.02,
                     entropy_threshold=0.8, entropy_temperature=0.6,
                     cons_start_epoch=5, cons_alpha=0.15,
                     cons_weak_sigma=0.02, cons_strong_sigma=0.08, cons_mask_ratio=0.2,
                     label_smoothing=0.0, use_entropy_minimization=False):
    """Linear Probe分類器を学習（VAT + Entropy Minimization + Consistency Regularization）

    - Epoch 0〜end: VAT（勾配ベース敵対的ノイズ）+ αウォームアップ
    - Epoch cons_start_epoch〜end: Consistency Regularization（embedding空間）
    - Epoch entropy_start_epoch〜end: Entropy Minimization追加（--no-emで無効化可能）

    Args:
        input_dim: 入力次元（768 or 774）
        label_smoothing: Label Smoothing係数（0.0-0.2推奨）
        use_entropy_minimization: EMを使用するかどうか
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    import math

    # Entropy正規化用定数（クラス数2の最大エントロピー）
    LOG_NUM_CLASSES = math.log(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")
    print(f"VAT: ε={vat_epsilon}, α={vat_alpha_start}→{vat_alpha_end} (全エポック)")
    print(f"Consistency: epoch {cons_start_epoch}〜, α={cons_alpha}, weak_σ={cons_weak_sigma}, strong_σ={cons_strong_sigma}, mask={cons_mask_ratio}")
    if use_entropy_minimization:
        print(f"Entropy Minimization: epoch {entropy_start_epoch}〜, α=0→{entropy_alpha_end}, τ={entropy_threshold}, T={entropy_temperature}")
    else:
        print("Entropy Minimization: DISABLED")
    if label_smoothing > 0:
        print(f"Label Smoothing: {label_smoothing}")

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/Val split (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"Class balance - Real: {(y_train == 0).sum()}, AI: {(y_train == 1).sum()}")

    # Tensors (long for CrossEntropyLoss)
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train.astype(int)).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val.astype(int)).to(device)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # Model - 動的次元対応
    print(f"Input dimension: {input_dim}")
    model = nn.Linear(input_dim, 2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
    nan_skip_count = 0
    em_loss_sum = 0.0
    em_count_sum = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        em_loss_sum = 0.0
        em_count_sum = 0

        # VAT用αのウォームアップ（全エポックで線形増加）
        vat_alpha = vat_alpha_start + (vat_alpha_end - vat_alpha_start) * (epoch / max(epochs - 1, 1))

        # Entropy Minimization用αのウォームアップ（中盤から線形増加）
        use_entropy = use_entropy_minimization and epoch >= entropy_start_epoch
        if use_entropy:
            entropy_epochs_total = epochs - entropy_start_epoch
            entropy_epoch_idx = epoch - entropy_start_epoch
            entropy_alpha = entropy_alpha_end * (entropy_epoch_idx / max(entropy_epochs_total - 1, 1))
        else:
            entropy_alpha = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # VAT: 勾配ベースの敵対的ノイズ
            # Step 1: ランダム方向のノイズで勾配を計算
            d = torch.randn_like(batch_x)
            d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
            d.requires_grad = True

            # 摂動を加えた予測
            logits_perturbed = model(batch_x + d * vat_epsilon)
            logits_clean = model(batch_x)

            # KLダイバージェンスで「最も不安定な方向」を見つける
            p_clean = F.softmax(logits_clean.detach(), dim=1)
            p_perturbed = F.log_softmax(logits_perturbed, dim=1)
            kl_loss = F.kl_div(p_perturbed, p_clean, reduction='batchmean')
            kl_loss.backward()

            # NaNチェック（勾配が壊れている場合はスキップ）
            if d.grad is None or torch.isnan(d.grad).any():
                optimizer.zero_grad(set_to_none=True)
                nan_skip_count += 1
                continue

            # Step 2: 最悪方向への敵対的ノイズを計算
            r_adv = d.grad / (d.grad.norm(dim=1, keepdim=True) + 1e-8) * vat_epsilon
            r_adv = r_adv.detach()

            optimizer.zero_grad(set_to_none=True)

            # メインロス（敵対的摂動を加えた入力で計算）
            logits = model(batch_x + r_adv)
            main_loss = criterion(logits, batch_y)

            # VATロス（クリーン vs 敵対的の一貫性）
            p_clean = F.softmax(model(batch_x).detach(), dim=1)
            p_adv = F.log_softmax(logits, dim=1)  # logitsを再利用（冗長性修正）
            vat_loss = F.kl_div(p_adv, p_clean, reduction='batchmean')

            # 合計ロス
            loss = main_loss + vat_alpha * vat_loss

            # Entropy Minimization（終盤、高信頼度サンプルのみ）
            if use_entropy:
                # 温度付きsoftmaxで分布を尖らせる
                probs = F.softmax(logits / entropy_temperature, dim=1)
                confidence, _ = torch.max(probs, dim=1)

                # 信頼度がτ以上のサンプルのみにEM適用
                high_conf_mask = confidence >= entropy_threshold
                n_high_conf = high_conf_mask.sum().item()
                em_count_sum += n_high_conf
                if n_high_conf > 0:
                    probs_high = probs[high_conf_mask]
                    # 正規化: log(C)で割って0.0〜1.0の範囲に
                    entropy_loss = -torch.mean(torch.sum(probs_high * torch.log(probs_high + 1e-8), dim=1)) / LOG_NUM_CLASSES
                    loss = loss + entropy_alpha * entropy_loss
                    em_loss_sum += entropy_loss.item()

            # Consistency Regularization（embedding空間での擬似augmentation）
            if epoch >= cons_start_epoch:
                # Weak augmentation: 軽いGaussian noise
                noise_weak = torch.randn_like(batch_x) * cons_weak_sigma
                x_weak = batch_x + noise_weak

                # Strong augmentation: 強いnoise + feature masking
                noise_strong = torch.randn_like(batch_x) * cons_strong_sigma
                mask = (torch.rand(batch_x.shape, device=device) > cons_mask_ratio).float()
                x_strong = (batch_x + noise_strong) * mask

                # Teacher (weak) の予測をdetach
                with torch.no_grad():
                    p_weak = F.softmax(model(x_weak), dim=1)

                # Student (strong) の予測
                p_strong = F.log_softmax(model(x_strong), dim=1)

                # Consistency Loss: KL(weak || strong)
                cons_loss = F.kl_div(p_strong, p_weak, reduction='batchmean')
                loss = loss + cons_alpha * cons_loss

            # NaNチェック（step()前に確認）
            if torch.isnan(loss):
                optimizer.zero_grad(set_to_none=True)
                nan_skip_count += 1
                continue

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        val_acc = correct / total if total > 0 else 0

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            cons_str = ", Cons" if epoch >= cons_start_epoch else ""
            if use_entropy:
                avg_em = em_loss_sum / max(len(train_loader), 1)
                em_str = f", EM={avg_em:.4f} ({em_count_sum})"
            else:
                em_str = ""
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc*100:.2f}% (VAT α={vat_alpha:.3f}{cons_str}{em_str})")

    print(f"\nBest Validation Accuracy: {best_acc*100:.2f}%")
    if nan_skip_count > 0:
        print(f"NaN skipped batches: {nan_skip_count}")

    # Save best model
    model.load_state_dict(best_state)
    return model, best_acc, input_dim


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Linear Probe classifier")
    parser.add_argument("--cls-only", action="store_true",
                        help="Use CLS only (768 dims) instead of CLS+patch_stats (774 dims)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: models/dinov3_classifier.pt)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing coefficient (0.0-0.2 recommended, default: 0.0)")
    parser.add_argument("--no-em", action="store_true",
                        help="Disable Entropy Minimization (EM)")
    args = parser.parse_args()

    use_patch_stats = not args.cls_only
    mode_str = "CLS + patch_stats (775 dims)" if use_patch_stats else "CLS only (768 dims)"

    print("=" * 50)
    print(f"Linear Probe Training: {mode_str}, {args.epochs} epochs")
    print("=" * 50)

    # 出力パス決定
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_PATH if use_patch_stats else OUTPUT_PATH.with_name("dinov3_classifier_cls_only.pt")
    # 既存のデフォルトモデルを上書きする場合のみバックアップ
    if output_path.exists() and args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(f"dinov3_classifier_backup_{timestamp}.pt")
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"Existing model backed up to {backup_path}")

    # Load embeddings
    X, y, input_dim = load_embeddings(use_patch_stats=use_patch_stats)

    # Train
    model, best_acc, input_dim = train_classifier(
        X, y, input_dim,
        epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        use_entropy_minimization=not args.no_em
    )

    # Save (checkpoint形式: バックエンドが checkpoint["classifier"] を期待)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier": model.state_dict(),
        "val_acc": best_acc,
        "input_dim": input_dim,
        "use_patch_stats": use_patch_stats
    }, output_path)
    print(f"\nModel saved to {output_path}")
    print(f"Validation accuracy: {best_acc*100:.2f}%")
    print(f"Input dimension: {input_dim}")


if __name__ == "__main__":
    main()
