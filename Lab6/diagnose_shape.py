"""
形狀混淆診斷工具：針對單一形狀條件 (cube/sphere/cylinder x 8色) 各生成幾張，
看模型對「形狀」這個訊號的遵循率有多高。
"""
import json
import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab6_conditional_ddpm import (
    ConditionalUNet, GaussianDiffusion, DDIMSampler, EvaluatorWrapper,
    load_object_map, labels_to_multihot, get_device, denorm,
    MODEL_PATH, IMAGE_SIZE, OUTPUT_DIR
)

def main():
    device = get_device()
    obj_map = load_object_map()
    idx2obj = {v: k for k, v in obj_map.items()}

    ck = torch.load(MODEL_PATH, map_location=device)
    base_ch = ck.get("base_ch", 128)
    timesteps = ck.get("timesteps", 1000)
    model = ConditionalUNet(base=base_ch).to(device)
    model.load_state_dict(ck.get("ema", ck["model"]))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=timesteps).to(device)
    ddim = DDIMSampler(diffusion, S=100, eta=0.0, cfg_scale=0.0)
    evaluator = EvaluatorWrapper()

    colors = ["gray","red","blue","green","brown","purple","cyan","yellow"]
    shapes = ["cube","sphere","cylinder"]

    # 針對每個形狀挑 3 個顏色測試（共 9 張單物體圖）
    test_labels = []
    for shape in shapes:
        for color in colors[:3]:
            test_labels.append([f"{color} {shape}"])

    conds = torch.stack([labels_to_multihot(l, obj_map) for l in test_labels]).to(device)
    imgs, _ = ddim.sample(model, conds, (conds.size(0), 3, IMAGE_SIZE, IMAGE_SIZE), device)
    imgs_cpu = denorm(imgs).cpu()

    save_image(make_grid(imgs_cpu, nrow=3, padding=2), OUTPUT_DIR/"shape_diagnostic.png")

    print(f"{'目標':<20} {'形狀正確?':<10} {'顏色正確?':<10}")
    shape_correct, color_correct = 0, 0
    for i, label in enumerate(test_labels):
        with torch.no_grad():
            out = evaluator.model(evaluator.norm(imgs_cpu[i:i+1].to(device)))[0].cpu()
        pred_idx = out.topk(1).indices.item()
        pred_name = idx2obj[pred_idx]
        target_color, target_shape = label[0].split()
        pred_color, pred_shape = pred_name.split()
        s_ok = "✓" if pred_shape == target_shape else "✗"
        c_ok = "✓" if pred_color == target_color else "✗"
        if pred_shape == target_shape: shape_correct += 1
        if pred_color == target_color: color_correct += 1
        print(f"{label[0]:<20} {s_ok:<10} {c_ok:<10} (預測={pred_name})")

    n = len(test_labels)
    print(f"\n形狀正確率: {shape_correct}/{n} = {shape_correct/n:.2%}")
    print(f"顏色正確率: {color_correct}/{n} = {color_correct/n:.2%}")
    print(f"圖片已存: shape_diagnostic.png")

if __name__ == "__main__":
    main()
