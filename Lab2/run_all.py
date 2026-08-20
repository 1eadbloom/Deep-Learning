"""
Lab2 一鍵流程：缺資料的話先下載，接著訓練、驗證，最後對測試集推論並輸出 CSV。

請在「專案根目錄」執行（也就是看得到 src/、dataset/ 的那一層）。

使用前請先裝依賴：
    pip install -r requirements.txt

資料怎麼切（與作業規定一致）：
    訓練／驗證名單都只來自 annotations/trainval.txt，在程式裡做成約 8:2（隨機但種子固定 42）。
    官方的 test.txt 只用在推論，不會參與訓練或驗證。
"""

import argparse
import os
import subprocess
import sys


# --- 路徑與常數 ---

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
DATA_ROOT = os.path.join(ROOT, "dataset", "oxford-iiit-pet")


# --- 子程序呼叫 ---


def run(cmd):
    """印出指令並在專案根目錄執行（方便助教重現）。"""
    print("\n" + "=" * 70)
    print(">>>", " ".join(cmd))
    print("=" * 70)
    subprocess.check_call(cmd, cwd=ROOT)


def dataset_ready():
    """檢查影像與 trimap 是否都已下載完成。"""
    return os.path.isdir(os.path.join(DATA_ROOT, "images")) and os.path.isdir(
        os.path.join(DATA_ROOT, "annotations", "trimaps")
    )


# --- 主流程 ---


def main():
    parser = argparse.ArgumentParser(
        description=(
            "從頭跑完 Lab2：可選是否下載資料，然後對每個模型依序 "
            "訓練 -> 驗證集評估 -> 測試集推論 -> 產生 submission_*.csv。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例：\n"
            "  python run_all.py\n"
            "  python run_all.py --epochs 50 --skip_download\n"
            "  python run_all.py --models unet\n"
            "\n"
            f"資料根目錄：{DATA_ROOT}\n"
            "先裝套件：pip install -r requirements.txt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="每個模型要訓練幾個 epoch（預設 50）",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="跳過下載；若本機沒資料夾就直接報錯",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "resnet34_unet"],
        choices=["unet", "resnet34_unet"],
        help="要跑哪幾個模型（預設兩個都跑）",
    )
    args = parser.parse_args()

    # 必要時先下載資料集
    if not args.skip_download and not dataset_ready():
        run([PYTHON, "src/download_dataset.py"])
    elif not dataset_ready():
        raise FileNotFoundError(
            f"找不到資料：{DATA_ROOT}。請先執行：python src/download_dataset.py"
        )

    # 各模型的訓練超參數（與作業建議一致）
    configs = {
        "unet": {"batch_size": 16, "lr": "1e-3"},
        "resnet34_unet": {"batch_size": 8, "lr": "5e-4"},
    }

    for model in args.models:
        cfg = configs[model]
        ckpt = os.path.join("saved_models", f"{model}_best.pth")

        # 1) 訓練
        run(
            [
                PYTHON,
                "src/train.py",
                "--model",
                model,
                "--data_root",
                DATA_ROOT,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(cfg["batch_size"]),
                "--lr",
                cfg["lr"],
                "--save_dir",
                "saved_models",
            ]
        )

        # 2) 驗證集評估（有真值）
        run(
            [
                PYTHON,
                "src/evaluate.py",
                "--model",
                model,
                "--checkpoint",
                ckpt,
                "--data_root",
                DATA_ROOT,
                "--split",
                "val",
            ]
        )

        # 3) 測試集推論 → Kaggle CSV（測試邏輯只在 inference.py）
        run(
            [
                PYTHON,
                "src/inference.py",
                "--model",
                model,
                "--checkpoint",
                ckpt,
                "--data_root",
                DATA_ROOT,
                "--output",
                f"submission_{model}.csv",
            ]
        )

    print("\n全部跑完了。")
    print("  權重檔在：saved_models/")
    print("  上傳用 CSV：submission_unet.csv、submission_resnet34_unet.csv")


if __name__ == "__main__":
    main()
