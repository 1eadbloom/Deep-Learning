"""
compare_models.py
-----------------
Generates the three images referenced in README.md:

    images/model_comparison.png
    images/unet_training_curves.png
    images/resnet34_unet_training_curves.png

Run from the project root:

    python compare_models.py

Requirements:
    saved_models/unet_best.pth
    saved_models/resnet34_unet_best.pth
    saved_models/unet_log.json
    saved_models/resnet34_unet_log.json
    dataset/oxford-iiit-pet/
"""

import os
import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import load_dataset

# ── config ───────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "dataset/oxford-iiit-pet"
SAVE_DIR  = "saved_models"
OUT_DIR   = "images"
N_SHOW    = 6
SEED      = 42
THRESHOLD = 0.5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

os.makedirs(OUT_DIR, exist_ok=True)


# ── font: use English-safe backend, no CJK needed ────────────────────────────
# All labels are written in English so Windows font issues are avoided entirely.
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.unicode_minus": False,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def denorm(tensor):
    img = tensor.numpy().transpose(1, 2, 0)
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def dice(pred, gt, eps=1e-6):
    return (2 * (pred * gt).sum() + eps) / (pred.sum() + gt.sum() + eps)


def load_model(model_fn, ckpt_path):
    model = model_fn().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    vd    = ckpt.get("val_dice", float("nan"))
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded {ckpt_path}  (epoch {epoch}, val_dice={vd:.4f})")
    return model, vd


# ── 1. model_comparison.png ───────────────────────────────────────────────────

def make_comparison_grid():
    unet_ckpt   = os.path.join(SAVE_DIR, "unet_best.pth")
    resnet_ckpt = os.path.join(SAVE_DIR, "resnet34_unet_best.pth")
    if not os.path.exists(unet_ckpt) or not os.path.exists(resnet_ckpt):
        print("[SKIP] model_comparison.png — checkpoint(s) missing"); return

    print("\n[1/3] Generating model_comparison.png ...")
    unet,   u_vd = load_model(lambda: UNet(in_channels=3, out_channels=1),          unet_ckpt)
    resnet, r_vd = load_model(lambda: ResNet34UNet(in_channels=3, out_channels=1),  resnet_ckpt)

    val_ds = load_dataset(DATA_ROOT, split="val", augment=False)
    idxs   = random.Random(SEED).sample(range(len(val_ds)), N_SHOW)

    fig, axes = plt.subplots(N_SHOW, 4, figsize=(16, N_SHOW * 3.5))
    fig.suptitle(
        f"Validation Set — UNet (val Dice {u_vd:.4f})  vs  "
        f"ResNet34-UNet (val Dice {r_vd:.4f})",
        fontsize=13, y=1.01
    )
    for ax, title in zip(axes[0],
                         ["Input Image", "Ground Truth", "UNet", "ResNet34-UNet"]):
        ax.set_title(title, fontsize=12, fontweight="bold")

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            img_t, mask_t = val_ds[idx]
            x   = img_t.unsqueeze(0).to(DEVICE)
            p_u = torch.sigmoid(unet(x))[0, 0].cpu().numpy()
            p_r = torch.sigmoid(resnet(x))[0, 0].cpu().numpy()

            img = denorm(img_t)
            gt  = mask_t[0].numpy()
            u_b = (p_u > THRESHOLD).astype(float)
            r_b = (p_r > THRESHOLD).astype(float)

            for ax, data, cmap, label in zip(
                axes[row],
                [img,  gt,     u_b,                    r_b],
                [None, "gray", "gray",                  "gray"],
                ["",   "",     f"Dice {dice(u_b,gt):.3f}", f"Dice {dice(r_b,gt):.3f}"]
            ):
                ax.imshow(data, cmap=cmap)
                if label:
                    ax.set_xlabel(label, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── 2 & 3. training curve plots ───────────────────────────────────────────────

def make_curve_plot(model_name, display_name):
    log_path = os.path.join(SAVE_DIR, f"{model_name}_log.json")
    out_path = os.path.join(OUT_DIR,   f"{model_name}_training_curves.png")

    if not os.path.exists(log_path):
        print(f"[SKIP] {model_name}_training_curves.png — {log_path} not found")
        return

    with open(log_path) as f:
        log = json.load(f)

    train_loss = log["train_loss"]
    val_loss   = log["val_loss"]
    val_dice   = log["val_dice"]

    # ── smooth the noisy val_dice curve with a 5-epoch rolling average ──
    def smooth(arr, w=5):
        out = []
        for i in range(len(arr)):
            lo = max(0, i - w // 2)
            hi = min(len(arr), i + w // 2 + 1)
            out.append(float(np.mean(arr[lo:hi])))
        return out

    val_dice_raw    = val_dice
    val_dice_smooth = smooth(val_dice, w=5)
    epochs = range(1, len(train_loss) + 1)

    best_epoch = int(np.argmax(val_dice_raw)) + 1
    best_dice  = max(val_dice_raw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{display_name} — Training Curves", fontsize=13)

    # Loss plot
    ax1.plot(epochs, train_loss, "b-o", ms=3, lw=1.5, label="Train Loss")
    ax1.plot(epochs, val_loss,   "r-o", ms=3, lw=1.5, label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Dice plot — raw (faint) + smoothed (bold)
    ax2.plot(epochs, val_dice_raw,    color="green", alpha=0.25, lw=1,   label="_nolegend_")
    ax2.plot(epochs, val_dice_smooth, color="green", lw=2,       label="Val Dice (smoothed)")
    ax2.axhline(0.85, color="orange", ls="--", lw=1.5, label="Baseline 0.85")
    ax2.set(xlabel="Epoch", ylabel="Dice Score",
            title="Validation Dice", ylim=(0.4, 1.0))
    ax2.legend(); ax2.grid(alpha=0.3)

    # Annotate best point
    ax2.annotate(
        f"Best: {best_dice:.4f} (epoch {best_epoch})",
        xy=(best_epoch, best_dice),
        xytext=(best_epoch + 3, best_dice - 0.05),
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
        fontsize=9, color="darkgreen"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    make_comparison_grid()

    print("\n[2/3] Generating unet_training_curves.png ...")
    make_curve_plot("unet", "UNet")

    print("\n[3/3] Generating resnet34_unet_training_curves.png ...")
    make_curve_plot("resnet34_unet", "ResNet34-UNet")

    print("\nDone. Files written to images/")
