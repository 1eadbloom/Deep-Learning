"""
對官方測試清單（annotations/test.txt）做推論，輸出 Kaggle 繳交用 CSV。

CSV 欄位：
    image_id：影像檔名（不含副檔名）
    encoded_mask：前景遮罩的 RLE 字串（欄優先展開、1-based 起跑；
                  若該張全背景則為空字串）

請在專案根目錄執行。

環境準備：
    pip install -r requirements.txt

範例：
    python src/inference.py --model unet --checkpoint saved_models/unet_best.pth \
        --data_root dataset/oxford-iiit-pet --output submission_unet.csv

修改紀錄：
    v2 — 加入水平翻轉 TTA（Test Time Augmentation），
         不需重新訓練即可提升推論穩定性與 Dice Score。
         preprocess_image 解析度同步更新為 384×384。
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import OxfordPetDataset
from utils import get_device, load_checkpoint


# --- Kaggle 繳交用的行程長度編碼（RLE）---


def mask_to_rle(mask):
    """
    把 H×W 的 0/1 二元遮罩編成 RLE 字串。
    像素順序為「先走欄」（Fortran order），索引從 1 開始。
    若整張都是 0 則回傳空字串。
    """
    mask_flat = mask.flatten(order="F").astype(np.uint8)
    if mask_flat.max() == 0:
        return ""

    padded  = np.concatenate([[0], mask_flat, [0]])
    diffs   = np.diff(padded.astype(int))
    starts  = np.where(diffs ==  1)[0] + 1
    ends    = np.where(diffs == -1)[0] + 1
    lengths = ends - starts

    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


# --- 單張影像前處理（與訓練時一致，解析度 384）---


def preprocess_image(image, img_size=(384, 384)):
    """把單張 PIL 影像轉成模型輸入張量（384×384，ImageNet 正規化）。"""
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR

    image  = image.resize(img_size, resample)
    img_np = np.array(image, dtype=np.float32) / 255.0
    mean   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float()


# --- TTA 推論：水平翻轉取機率平均 ---


@torch.no_grad()
def _predict_with_tta(model, x, device):
    """
    對單一 batch 做水平翻轉 TTA。

    步驟：
      1. 原圖推論 → sigmoid 機率
      2. 水平翻轉後推論 → sigmoid 機率 → 翻轉回來對齊空間
      3. 兩者機率平均後二值化
    """
    x      = x.to(device)
    x_flip = torch.flip(x, dims=[-1])          # 沿 W 軸翻轉

    prob      = torch.sigmoid(model(x))
    prob_flip = torch.sigmoid(model(x_flip))
    prob_flip = torch.flip(prob_flip, dims=[-1])  # 翻回來對齊

    return (prob + prob_flip) / 2               # 機率平均


# --- 批次推論（DataLoader，含 TTA）---


@torch.no_grad()
def run_inference(model, loader, device, threshold=0.5, use_tta=True):
    """
    跑完整個 loader，回傳每張預測的 H×W 二元遮罩（uint8，0 或 1）。
    use_tta=True 時使用水平翻轉 TTA，可提升邊界預測穩定性。
    """
    model.eval()
    results = []

    for images, masks in loader:
        if use_tta:
            avg_prob = _predict_with_tta(model, images, device)
        else:
            images   = images.to(device)
            avg_prob = torch.sigmoid(model(images))

        # 若輸出尺寸與 mask 不一致，對齊到 mask 大小
        if avg_prob.shape != masks.shape:
            avg_prob = torch.nn.functional.interpolate(
                avg_prob, size=masks.shape[2:],
                mode="bilinear", align_corners=False
            )

        pred_np = avg_prob.cpu().numpy()
        for i in range(pred_np.shape[0]):
            results.append((pred_np[i, 0] > threshold).astype(np.uint8))

    return results


# --- 依檔名清單逐張推論（含 TTA）---


@torch.no_grad()
def run_inference_from_name_list(
    model, images_dir, image_names, device, threshold=0.5, use_tta=True
):
    """依指定檔名清單逐張讀圖推論（含水平翻轉 TTA）。"""
    model.eval()
    results = []

    for name in image_names:
        image_path = os.path.join(images_dir, f"{name}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到測試影像：{image_path}")

        image  = Image.open(image_path).convert("RGB")
        tensor = preprocess_image(image).unsqueeze(0)  # (1, 3, H, W)

        if use_tta:
            avg_prob = _predict_with_tta(model, tensor, device)
        else:
            avg_prob = torch.sigmoid(model(tensor.to(device)))

        pred_bin = (avg_prob[0, 0].cpu().numpy() > threshold).astype(np.uint8)
        results.append(pred_bin)

    return results


# --- 產生繳交 CSV ---


def generate_submission(
    model, data_root, device, threshold, output_csv,
    model_name, test_list=None, use_tta=True
):
    """讀測試集、推論並寫成繳交用 CSV（image_id, encoded_mask）。"""
    test_dataset = OxfordPetDataset(root=data_root, split="test", transform=False)
    tta_note     = "（含 TTA）" if use_tta else ""

    if test_list is not None:
        with open(test_list, "r", encoding="utf-8") as f:
            target_names = [line.strip() for line in f if line.strip()]
        print(f"使用指定名單推論{tta_note}，共 {len(target_names)} 張…")
        images_dir = os.path.join(data_root, "images")
        pred_masks = run_inference_from_name_list(
            model, images_dir, target_names, device,
            threshold=threshold, use_tta=use_tta
        )
    else:
        target_names = test_dataset.samples
        num_workers  = 0 if os.name == "nt" else 4
        test_loader  = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"正在對 {len(test_dataset)} 張測試影像推論{tta_note}…")
        pred_masks = run_inference(
            model, test_loader, device,
            threshold=threshold, use_tta=use_tta
        )

    rows = [
        {"image_id": name, "encoded_mask": mask_to_rle(mask)}
        for name, mask in zip(target_names, pred_masks)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"已輸出：{output_csv}")
    print(f"筆數：{len(df)}")
    print(f"全背景（encoded_mask 為空）張數：{(df['encoded_mask'] == '').sum()}")
    return df


# --- 命令列參數 ---


def get_args():
    parser = argparse.ArgumentParser(
        description="載入權重後對官方測試影像推論，輸出 Kaggle CSV。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "請在專案根目錄執行。先安裝：pip install -r requirements.txt\n"
            "若未指定 --output，預設檔名為 submission_<model>.csv\n"
            "上傳前請對照該場競賽公布的 RLE／格式說明。"
        ),
    )
    parser.add_argument(
        "--model", type=str, default="unet",
        choices=["unet", "resnet34_unet"], help="模型種類",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="訓練好的 .pth",
    )
    parser.add_argument(
        "--data_root", type=str, default="dataset/oxford-iiit-pet",
        help="資料集根目錄",
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="預測二值化門檻")
    parser.add_argument(
        "--output", type=str, default=None,
        help="輸出 CSV 路徑（預設 submission_{model}.csv）",
    )
    parser.add_argument(
        "--test_list", type=str, default=None,
        help="可選：指定競賽測試名單 txt（每行一個 image_id）",
    )
    parser.add_argument(
        "--no_tta", action="store_true",
        help="停用 TTA（預設開啟水平翻轉 TTA）",
    )
    return parser.parse_args()


# --- 程式進入點 ---


if __name__ == "__main__":
    args   = get_args()
    device = get_device()
    print(f"使用裝置：{device}")

    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1, base_features=64)
    else:
        model = ResNet34UNet(in_channels=3, out_channels=1)

    model = model.to(device)
    load_checkpoint(model, args.checkpoint, device)

    output_csv = args.output or f"submission_{args.model}.csv"
    use_tta    = not args.no_tta

    generate_submission(
        model=model,
        data_root=args.data_root,
        device=device,
        threshold=args.threshold,
        output_csv=output_csv,
        model_name=args.model,
        test_list=args.test_list,
        use_tta=use_tta,
    )
