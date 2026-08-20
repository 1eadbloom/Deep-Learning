"""
二元語意分割訓練腳本（Oxford-IIIT Pet：trimap 轉二元遮罩）。

支援 UNet 與 ResNet34+UNet。請在「專案根目錄」執行（與 src/ 同層）。

環境準備：
    pip install -r requirements.txt

範例：
    python src/train.py --model unet --epochs 50 --batch_size 16 --lr 1e-3 \\
        --data_root dataset/oxford-iiit-pet
    python src/train.py --model resnet34_unet --epochs 50 --batch_size 8 --lr 5e-4 \\
        --data_root dataset/oxford-iiit-pet

訓練／驗證切分請參考 oxford_pet.py（僅從 trainval.txt 留出驗證）。
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import load_dataset
from utils import (
    BCEDiceLoss,
    dice_score,
    get_device,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    plot_training_curves,
)


# --- 單輪訓練 ---


def train_one_epoch(model, loader, optimizer, criterion, device):
    """跑完一個 epoch 的訓練：前向、算 loss、反傳、更新權重。"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc="訓練", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # 輸出尺寸若與標註差一點點，用雙線性對齊（常見於 padding 造成的誤差）
        if logits.shape != masks.shape:
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[2:], mode="bilinear", align_corners=False
            )

        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            d = dice_score(logits, masks)

        total_loss += loss.item()
        total_dice += d.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d.item():.4f}")

    return total_loss / len(loader), total_dice / len(loader)


# --- 驗證集評估（有真值，非 Kaggle test）---


@torch.no_grad()
def validate(model, loader, criterion, device):
    """在驗證集上算平均 loss 與 Dice，用來挑最佳 checkpoint。"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="驗證", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        if logits.shape != masks.shape:
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[2:], mode="bilinear", align_corners=False
            )

        loss = criterion(logits, masks)
        d = dice_score(logits, masks)

        total_loss += loss.item()
        total_dice += d.item()

    return total_loss / len(loader), total_dice / len(loader)


# --- 主訓練流程 ---


def train(args):
    device = get_device()
    print(f"使用裝置：{device}")

    # 載入 train／val（皆來自 trainval.txt，見 oxford_pet.py）
    data_root = args.data_root
    train_dataset = load_dataset(data_root, split="train", augment=True)
    val_dataset = load_dataset(data_root, split="val", augment=False)

    print(f"訓練張數：{len(train_dataset)} ｜ 驗證張數：{len(val_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # 依參數建立 UNet 或 ResNet34+UNet
    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1, base_features=64)
        model_name = "unet"
    elif args.model == "resnet34_unet":
        model = ResNet34UNet(in_channels=3, out_channels=1)
        model_name = "resnet34_unet"
    else:
        raise ValueError(f"不支援的模型：{args.model}")

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"模型：{model_name}")
    print(f"參數總數：{total_params:,} ｜ 可訓練：{trainable_params:,}")

    # 損失、優化器、學習率排程
    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_dice = load_checkpoint(model, args.resume, device, optimizer)

    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, f"{model_name}_best.pth")

    train_losses, val_losses, val_dices = [], [], []

    # 逐 epoch 訓練並記錄曲線
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"第 {epoch + 1}/{args.epochs} 輪  目前學習率：{scheduler.get_last_lr()[0]:.6f}")
        print("=" * 60)

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(
            f"\n本輪摘要 — "
            f"訓練 Loss：{train_loss:.4f}  訓練 Dice：{train_dice:.4f} ｜ "
            f"驗證 Loss：{val_loss:.4f}  驗證 Dice：{val_dice:.4f}"
        )

        # 驗證 Dice 變好就存最佳權重
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch + 1, val_dice, best_ckpt)
            print(f"更新最佳模型，驗證 Dice：{val_dice:.4f}")

        # 每 10 輪多存一份，方便之後回溯
        if (epoch + 1) % 10 == 0:
            periodic_ckpt = os.path.join(
                args.save_dir, f"{model_name}_epoch{epoch + 1}.pth"
            )
            save_checkpoint(model, optimizer, epoch + 1, val_dice, periodic_ckpt)

    print(f"\n訓練結束。最佳驗證 Dice：{best_dice:.4f}")

    curves_path = os.path.join(args.save_dir, f"{model_name}_training_curves.png")
    plot_training_curves(train_losses, val_losses, val_dices, save_path=curves_path)

    # 儲存訓練紀錄供 compare_models.py 繪製曲線圖使用
    import json
    log = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "val_dice":   val_dices,
    }
    log_path = os.path.join(args.save_dir, f"{model_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f)
    print(f"訓練紀錄已儲存：{log_path}")


# --- 命令列參數 ---


def get_args():
    parser = argparse.ArgumentParser(
        description="在 Oxford-IIIT Pet 上訓練 UNet 或 ResNet34+UNet（二元分割）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "請在專案根目錄執行。先安裝：pip install -r requirements.txt\n"
            "最佳權重會存成：<save_dir>/<model>_best.pth"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "resnet34_unet"],
        help="要訓練的架構",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset/oxford-iiit-pet",
        help="資料集根目錄（底下要有 images/ 與 annotations/）",
    )
    parser.add_argument("--epochs", type=int, default=80, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小（384 解析度建議 unet:8, resnet34:4）")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始學習率")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0 if os.name == "nt" else 4,
        help="DataLoader 背景工作數（Windows 通常預設 0 ）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="checkpoint 輸出目錄",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="從既有 checkpoint 接續訓練（路徑）",
    )
    return parser.parse_args()


# --- 程式進入點 ---


if __name__ == "__main__":
    args = get_args()
    train(args)
