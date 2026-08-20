"""
Oxford-IIIT Pet 資料：把 trimap 轉成二元遮罩後做語意分割。

切割資料：
    ‧ train／val：都只從 annotations/trainval.txt 抽出，程式裡做成約 8:2（種子固定 42）。
    ‧ test：用 annotations/test.txt，僅供 inference.py 使用，不要拿來訓練或選模型。

Trimap 意義：1＝前景、2＝背景、3＝邊界。邊界依規定一律當背景（輸出 0）。

修改紀錄：
    v2 — 修正 _augment() 提前 return 造成 color jitter 永遠不執行的 bug。
         預設解析度從 256×256 提升至 384×384 以保留更多邊界細節。
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop


# --- 訓練／驗證切分設定（固定種子，可重現）---

_VAL_RATIO  = 0.2
_SPLIT_SEED = 42


# --- 讀取官方名單檔 ---


def _read_split_file(list_file):
    """讀 trainval.txt 或 test.txt，每行第一欄為影像檔名（不含副檔名）。"""
    samples = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                samples.append(parts[0])
    return samples


def _split_trainval(samples, val_ratio=_VAL_RATIO, seed=_SPLIT_SEED):
    """把 trainval 清單隨機切成訓練集與驗證集（不用 test.txt 當 val）。"""
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    val_samples   = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    return train_samples, val_samples


# --- PyTorch Dataset ---


class OxfordPetDataset(Dataset):
    """
    讀取 Oxford-IIIT Pet，輸出二元分割用的影像與遮罩。

    Trimap：1＝前景、2＝背景、3＝邊界。
    轉成二元後：前景為 1，其餘（含邊界）為 0。

    資料切分：
      ‧ train／val：僅來自 trainval.txt
      ‧ test：官方 test.txt，推論用；訓練階段不使用此清單
    """

    def __init__(self, root, split="train", transform=None, img_size=(384, 384)):
        # ↑ 解析度從 256 提升至 384，保留更多寵物邊界細節
        self.root     = root
        self.split    = split
        self.transform = transform
        self.img_size  = img_size

        self.images_dir = os.path.join(root, "images")
        self.masks_dir  = os.path.join(root, "annotations", "trimaps")
        split_dir       = os.path.join(root, "annotations")

        trainval_file = os.path.join(split_dir, "trainval.txt")
        test_file     = os.path.join(split_dir, "test.txt")

        if split in ("train", "val"):
            all_trainval = _read_split_file(trainval_file)
            train_samples, val_samples = _split_trainval(all_trainval)
            self.samples = train_samples if split == "train" else val_samples
        elif split == "test":
            self.samples = _read_split_file(test_file)
        else:
            raise ValueError(f"不認得的 split：{split}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, name):
        path = os.path.join(self.images_dir, name + ".jpg")
        return Image.open(path).convert("RGB")

    def _load_mask(self, name):
        path = os.path.join(self.masks_dir, name + ".png")
        return Image.open(path)

    @staticmethod
    def _convert_trimap_to_binary(mask_np):
        """Trimap 轉二元：只有標籤 1 當前景；2、3（含邊界）都算背景。"""
        binary = np.zeros_like(mask_np, dtype=np.float32)
        binary[mask_np == 1] = 1.0
        return binary

    def _augment(self, image, mask):
        """
        訓練用增強：影像與遮罩用同一組幾何變換，顏色只動影像。

        修正：移除 RandomResizedCrop 分支裡的提前 return，
        確保 color jitter 在所有路徑下都有機會被套用。
        """
        # 1. 水平翻轉
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # 2. 隨機旋轉 ±15°
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(
                image, angle, interpolation=TF.InterpolationMode.BILINEAR
            )
            mask = TF.rotate(
                mask, angle, interpolation=TF.InterpolationMode.NEAREST
            )

        # 3. 隨機裁切並縮放回目標尺寸
        #    【修正】：原版在此提前 return，導致後面的 color jitter 永遠不執行。
        #              改為：裁切後繼續往下執行，最後統一 return。
        if random.random() > 0.5:
            i, j, h, w = RandomResizedCrop.get_params(
                image, scale=(0.85, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(
                image, i, j, h, w, self.img_size,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.resized_crop(
                mask, i, j, h, w, self.img_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )

        # 4. 顏色抖動（只動影像，不動遮罩）
        #    【修正】：原版因上方提前 return 而常常被跳過，現在每條路徑都會跑到這裡。
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            image = TF.adjust_contrast(image,   random.uniform(0.7, 1.3))
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        return image, mask   # 唯一的 return，確保所有增強步驟都有機會執行

    def __getitem__(self, idx):
        """回傳 (影像張量, 遮罩張量)，形狀分別為 (3,H,W) 與 (1,H,W)。"""
        name  = self.samples[idx]
        image = self._load_image(name)
        mask  = self._load_mask(name)

        # 統一縮放到目標尺寸
        try:
            bilinear = Image.Resampling.BILINEAR
            nearest  = Image.Resampling.NEAREST
        except AttributeError:          # Pillow < 9.1 相容
            bilinear = Image.BILINEAR
            nearest  = Image.NEAREST

        image = image.resize(self.img_size, bilinear)
        mask  = mask.resize(self.img_size,  nearest)

        # 訓練時做增強
        if self.split == "train" and self.transform:
            image, mask = self._augment(image, mask)
            # 若 _augment 裡的裁切沒觸發，確保尺寸仍正確
            if image.size != (self.img_size[1], self.img_size[0]):
                image = image.resize(self.img_size, bilinear)
                mask  = mask.resize(self.img_size,  nearest)

        # Trimap → 二元遮罩
        mask_np      = np.array(mask, dtype=np.int32)
        binary_mask  = self._convert_trimap_to_binary(mask_np)

        # ImageNet 正規化
        img_np = np.array(image, dtype=np.float32) / 255.0
        mean   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_tensor  = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(binary_mask).float().unsqueeze(0)
        return img_tensor, mask_tensor


# --- 對外載入介面 ---


def load_dataset(data_root, split, augment=False):
    """載入指定 split；augment=True 時僅對 train 有意義。"""
    return OxfordPetDataset(
        root=data_root,
        split=split,
        transform=augment,
    )
