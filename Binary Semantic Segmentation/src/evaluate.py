"""
評估腳本：在「訓練」或「驗證」split 上逐張計算 Dice（有 trimap 真值才能算）。

這裡的 val 是從 trainval.txt 留出來的驗證集，非 Kaggle 官方測試集；
測試集沒公開標籤，若要產生繳交檔請用 inference.py。

請在專案根目錄執行。

環境準備：
    pip install -r requirements.txt

範例：
    python src/evaluate.py --model unet --checkpoint saved_models/unet_best.pth \\
        --data_root dataset/oxford-iiit-pet --split val
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import load_dataset
from utils import get_device, load_checkpoint, dice_score_numpy, visualize_predictions


# --- 在有標註的 split 上算 Dice ---


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, visualize=False):
    """逐張預測並與真值比對，回傳平均 Dice 與每張分數列表。"""
    model.eval()
    all_dices = []

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        if logits.shape != masks.shape:
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[2:], mode="bilinear", align_corners=False
            )

        pred_prob = torch.sigmoid(logits).cpu().numpy()
        true_masks = masks.cpu().numpy()

        # 逐張影像算 Dice（作業要求用整體平均，不是只挑最好的一張）
        for i in range(pred_prob.shape[0]):
            pred_bin = (pred_prob[i, 0] > threshold).astype(np.float32)
            true_bin = true_masks[i, 0].astype(np.float32)
            d = dice_score_numpy(pred_bin, true_bin)
            all_dices.append(d)

        if visualize and batch_idx == 0:
            visualize_predictions(images.cpu(), masks.cpu(), logits.cpu(), n=4)

    mean_dice = np.mean(all_dices)
    std_dice = np.std(all_dices)
    min_dice = np.min(all_dices)
    max_dice = np.max(all_dices)

    print(f"\n{'='*50}")
    print(f"評估結果（共 {len(all_dices)} 張）")
    print(f"{'='*50}")
    print(f"  平均 Dice： {mean_dice:.4f}")
    print(f"  標準差：   {std_dice:.4f}")
    print(f"  最小：     {min_dice:.4f}")
    print(f"  最大：     {max_dice:.4f}")
    print(f"  Dice > 0.85：{sum(d > 0.85 for d in all_dices)}/{len(all_dices)} 張")
    print(f"{'='*50}\n")

    return mean_dice, all_dices


# --- 命令列參數 ---


def get_args():
    parser = argparse.ArgumentParser(
        description="在有標註的 train／val split 上計算 Dice（來自 trimap 二元化）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "請在專案根目錄執行。先安裝：pip install -r requirements.txt\n"
            "官方測試集無標籤；測試影像請改用 inference.py 產生 CSV。"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "resnet34_unet"],
        help="模型種類",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="訓練好的 .pth 路徑",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset/oxford-iiit-pet",
        help="資料根目錄（含 images/、annotations/）",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0 if os.name == "nt" else 4,
        help="DataLoader 工作者數",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="預測二值化門檻")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help=(
            "val：從 trainval 留出的驗證集，適合選模型；"
            "train：其餘訓練子集，用來除錯或對照"
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="顯示第一批的預測對照圖",
    )
    return parser.parse_args()


# --- 程式進入點 ---


if __name__ == "__main__":
    args = get_args()
    device = get_device()
    print(f"使用裝置：{device}")

    # 載入權重並建立對應架構
    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1, base_features=64)
    else:
        model = ResNet34UNet(in_channels=3, out_channels=1)

    model = model.to(device)
    load_checkpoint(model, args.checkpoint, device)

    dataset = load_dataset(args.data_root, split=args.split, augment=False)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print(f"正在評估 split「{args.split}」，共 {len(dataset)} 張…")

    evaluate(model, loader, device, threshold=args.threshold, visualize=args.visualize)
