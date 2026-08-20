"""
下載 Oxford-IIIT Pet，放到 /dataset/oxford-iiit-pet/。

請先安裝必要套件：
    pip install -r requirements.txt

在專案根目錄執行：
    python src/download_dataset.py

下載完即可接 train.py、evaluate.py，或直接用 run_all.py。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torchvision.datasets import OxfordIIITPet


# --- 下載主流程 ---


def main():
    parser = argparse.ArgumentParser(
        description=(
            "透過 torchvision 下載 Oxford-IIIT Pet（trainval 與 test 清單），"
            "存放在 dataset/oxford-iiit-pet/。"
        ),
        epilog="請先執行：pip install -r requirements.txt",
    )
    parser.parse_args()

    # 專案根目錄下的 dataset/ 資料夾
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "dataset")
    os.makedirs(dataset_root, exist_ok=True)

    print(f"開始下載資料到 {dataset_root} …")
    OxfordIIITPet(root=dataset_root, split="trainval", download=True)
    OxfordIIITPet(root=dataset_root, split="test", download=True)
    print("下載完成。")
    print(f"  影像：{os.path.join(dataset_root, 'oxford-iiit-pet', 'images')}")
    print(f"  標註：{os.path.join(dataset_root, 'oxford-iiit-pet', 'annotations')}")


if __name__ == "__main__":
    main()
