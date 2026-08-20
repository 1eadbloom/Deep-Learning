"""
訓練／評估／推論共用的工具函式。

內容包含：
    ‧ Dice 指標（Tensor 與 NumPy 版）
    ‧ BCE + Dice 複合損失
    ‧ 預測對照圖與訓練曲線繪製
    ‧ 裝置選擇與 checkpoint 讀寫
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# --- 指標 ---


def dice_score(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    """
    對整個 batch 算 Dice，回傳平均後的標量 Tensor。

    pred_mask： (B, 1, H, W)，尚未 sigmoid 的 logits 也行（內部會 sigmoid）。
    true_mask： (B, 1, H, W)，0/1 真值。
    threshold：把預測機率二值化的門檻。
    eps：避免分母為 0。
    """
    pred_prob = torch.sigmoid(pred_mask)
    pred_bin = (pred_prob > threshold).float()
    true_bin = true_mask.float()

    intersection = (pred_bin * true_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + true_bin.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_score_numpy(pred_bin, true_bin, eps=1e-6):
    """NumPy 版 Dice，給評估時逐張聚合用。"""
    intersection = (pred_bin * true_bin).sum()
    union = pred_bin.sum() + true_bin.sum()
    return (2 * intersection + eps) / (union + eps)


# --- 損失 ---


class DiceLoss(nn.Module):
    """可微分的 Soft Dice Loss（對機率算 overlap）。"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE（含 logits）與 Dice Loss 各一半加總。
    BCE顧像素分類，Dice 拉近區域重疊；alpha 預設 0.5 表示兩者比重相同。
    """

    def __init__(self, alpha=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


# --- 視覺化 ---


def visualize_predictions(images, true_masks, pred_masks, n=4, save_path=None):
    """
    並排顯示：原圖、真值遮罩、預測遮罩。

    images：已做 ImageNet normalize 的 (B, 3, H, W)；會反標準化再畫。
    true_masks／pred_masks：(B, 1, H, W)，後者可為 logits。
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    n = min(n, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    pred_prob = torch.sigmoid(pred_masks).cpu().numpy()

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        gt = true_masks[i, 0].cpu().numpy()
        pred = (pred_prob[i, 0] > 0.5).astype(float)

        axes[i][0].imshow(img)
        axes[i][0].set_title("輸入影像")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt, cmap="gray")
        axes[i][1].set_title("真值遮罩")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred, cmap="gray")
        axes[i][2].set_title("模型預測")
        axes[i][2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"對照圖已存：{save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, val_dices, save_path=None):
    """畫訓練／驗證 loss 與驗證 Dice 曲線。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-o", label="訓練 Loss", markersize=4)
    ax1.plot(epochs, val_losses, "r-o", label="驗證 Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss 曲線（訓練／驗證）")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, "g-o", label="驗證 Dice", markersize=4)
    ax2.axhline(y=0.85, color="orange", linestyle="--", label="基準線（0.85）")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("驗證 Dice")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"訓練曲線已存：{save_path}")
    plt.close()


# --- 模型參數統計 ---


def count_parameters(model):
    """回傳（總參數量，可訓練參數量）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --- 運算裝置 ---


def get_device():
    """挑可用的運算裝置：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# --- Checkpoint 存取 ---


def save_checkpoint(model, optimizer, epoch, val_dice, path):
    """存訓練狀態（模型與優化器）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
        },
        path,
    )
    print(f"已存 checkpoint：{path}（第 {epoch} 輪，val_dice={val_dice:.4f}）")


def load_checkpoint(model, path, device, optimizer=None):
    """載入 checkpoint；若有給 optimizer 則一併還原。"""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    val_dice = checkpoint.get("val_dice", 0.0)
    print(f"已載入：{path}（第 {epoch} 輪，val_dice={val_dice:.4f}）")
    return epoch, val_dice
