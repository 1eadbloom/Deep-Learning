"""
分析工具：逐張影像評估準確率，找出哪些 label 組合表現差
使用方式：python analyze_accuracy.py
需要先執行過 --mode generate，images/test/ 和 images/new_test/ 資料夾要存在
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT      = Path(__file__).resolve().parent
FILE_DIR  = ROOT / "file"
IMG_DIR   = ROOT / "images"
NUM_CLASSES = 24

def load_object_map():
    with open(FILE_DIR/"objects.json", encoding="utf-8") as f:
        return json.load(f)

def labels_to_multihot(label_list, obj_map):
    v = torch.zeros(NUM_CLASSES)
    for n in label_list:
        if n in obj_map: v[obj_map[n]] = 1.0
    return v

class Evaluator:
    def __init__(self):
        import torchvision.models as tvm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(FILE_DIR/"checkpoint.pth", map_location=device)
        r = tvm.resnet18(weights=None)
        r.fc = nn.Sequential(nn.Linear(512,24), nn.Sigmoid())
        r.load_state_dict(ckpt["model"])
        self.model = r.to(device).eval()
        self.device = device
        self.norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    @torch.no_grad()
    def per_sample_acc(self, img, label):
        x = self.norm(img.unsqueeze(0).to(self.device))
        out = self.model(x)[0].cpu()
        k = int(label.sum().item())
        outi = out.topk(k).indices.tolist()
        li   = label.topk(k).indices.tolist()
        hit  = len(set(outi) & set(li))
        return hit / k, outi, li


def main():
    obj_map = load_object_map()
    evaluator = Evaluator()
    tfm = transforms.Compose([
        transforms.Resize((64,64)), transforms.ToTensor(),
    ])
    idx2obj = {v:k for k,v in obj_map.items()}

    for split in ["test", "new_test"]:
        json_path = FILE_DIR / f"{split}.json"
        with open(json_path, encoding="utf-8") as f:
            labels_list = json.load(f)

        print(f"\n{'='*70}\n{split}.json 逐筆分析\n{'='*70}")
        accs = []
        for i, label_names in enumerate(labels_list):
            img_path = IMG_DIR / split / f"{i}.png"
            if not img_path.exists():
                print(f"  [{i}] 找不到圖片: {img_path}")
                continue
            img = tfm(Image.open(img_path).convert("RGB"))
            label = labels_to_multihot(label_names, obj_map)
            acc, pred_idx, true_idx = evaluator.per_sample_acc(img, label)
            accs.append(acc)
            pred_names = [idx2obj[j] for j in pred_idx]
            mark = "✓" if acc == 1.0 else ("△" if acc > 0 else "✗")
            print(f"  [{i:2d}] {mark} acc={acc:.2f}  目標={label_names}  預測命中={pred_names}")

        single_idx = [i for i,l in enumerate(labels_list) if len(l)==1]
        multi_idx  = [i for i,l in enumerate(labels_list) if len(l)>1]
        if single_idx:
            single_acc = sum(accs[i] for i in single_idx) / len(single_idx)
            print(f"\n  單物體樣本（{len(single_idx)}筆）平均準確率: {single_acc:.4f}")
        if multi_idx:
            multi_acc = sum(accs[i] for i in multi_idx) / len(multi_idx)
            print(f"  多物體樣本（{len(multi_idx)}筆）平均準確率: {multi_acc:.4f}")
        print(f"  整體平均: {sum(accs)/len(accs):.4f}")


if __name__ == "__main__":
    main()
