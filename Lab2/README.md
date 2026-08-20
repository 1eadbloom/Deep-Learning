# Binary Semantic Segmentation — Oxford-IIIT Pet Dataset

> \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*Deep Learning Lab2\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\* — NYCU, Spring 2026
> UNet and ResNet34-UNet for pixel-level pet segmentation, trained from scratch with PyTorch

\---

## Results

|Model|Val Dice|Train Dice|Train/Val Gap|
|-|-|-|-|
|UNet|**0.8908**|0.9234|0.033|
|ResNet34-UNet|**0.9126**|0.9392|0.027|

> Baseline threshold: Dice > 0.85. Both models exceed baseline.
> ResNet34-UNet shows a smaller train/val gap (0.027), indicating better generalisation.

### Prediction Examples

> Sample predictions from the validation set. Each row: original image | ground truth mask | UNet prediction | ResNet34-UNet prediction.

!\[Model comparison](images/model\_comparison.png)

### UNet — Training Curves

!\[UNet training curves](images/unet\_training\_curves.png)

### ResNet34-UNet — Training Curves

!\[ResNet34-UNet training curves](images/resnet34\_unet\_training\_curves.png)

\---

## Problem Statement

Given a pet image from the Oxford-IIIT Pet Dataset, classify every pixel as either **foreground** (the pet) or **background**.

The dataset provides trimap annotations with three labels:

|Label|Meaning|Binary mapping|
|-|-|-|
|`1`|Pet body|→ Foreground (`1`)|
|`2`|Background|→ Background (`0`)|
|`3`|Boundary (ambiguous)|→ Background (`0`)|

Boundary pixels are treated as background per the lab specification. This mapping is critical — incorrect handling of label `3` was a key source of low scores in early submissions.

\---

## Key Finding: Score Was Low Due to Implementation Bugs, Not Architecture

Early submissions scored **Dice < 0.80** on Kaggle, below the 0.85 baseline. The root cause was not the model architecture — it was three implementation issues in the training pipeline.

### Bug 1 — Augmentation Logic (Highest Impact)

`oxford\\\\\\\\\\\\\\\_pet.py`'s `\\\\\\\\\\\\\\\_augment()` had a premature `return` inside the RandomResizedCrop branch, causing color jitter to **never execute** on the code path where cropping was applied:

```python
# Before (buggy): color jitter silently skipped \\\\\\\\\\\\\\\~50% of the time
def \\\\\\\\\\\\\\\_augment(self, image, mask):
    if random.random() > 0.5:
        # ... random crop ...
        return image, mask          # ← early return; color jitter never reached

    if random.random() > 0.5:      # ← this block often unreachable
        image = TF.adjust\\\\\\\\\\\\\\\_brightness(image, random.uniform(0.7, 1.3))
        ...
```

```python
# After (fixed): all augmentation paths execute
def \\\\\\\\\\\\\\\_augment(self, image, mask):
    if random.random() > 0.5:
        # ... random crop ...
        # no early return — execution continues below

    if random.random() > 0.5:      # ← now reachable on every path
        image = TF.adjust\\\\\\\\\\\\\\\_brightness(image, random.uniform(0.7, 1.3))
        ...

    return image, mask              # single return at the end
```

Incomplete augmentation reduced training sample diversity, limiting generalisation.

### Bug 2 — Input Resolution Too Low

All images were resized to 256×256. Oxford-IIIT Pet images are natively 300–500px; compressing to 256 discards fine boundary detail that directly affects Dice Score.

||Before|After|
|-|-|-|
|Input resolution|256×256|**384×384**|

### Bug 3 — Insufficient Training Epochs

50 epochs with CosineAnnealingLR terminates too early; the model had not converged.

||Before|After|
|-|-|-|
|Epochs|50|**80**|

### Additional: TTA at Inference

Horizontal-flip Test Time Augmentation was added to `inference.py`. No retraining required — it averages the predicted probability maps from the original and flipped image:

```python
# Before: single-pass inference
prob = torch.sigmoid(model(x))

# After: TTA — horizontal flip average
prob      = torch.sigmoid(model(x))
prob\\\\\\\\\\\\\\\_flip = torch.sigmoid(model(torch.flip(x, dims=\\\\\\\\\\\\\\\[-1])))
prob\\\\\\\\\\\\\\\_flip = torch.flip(prob\\\\\\\\\\\\\\\_flip, dims=\\\\\\\\\\\\\\\[-1])   # un-flip to align spatially
avg\\\\\\\\\\\\\\\_prob  = (prob + prob\\\\\\\\\\\\\\\_flip) / 2
```

### Before vs After

||Before|After|
|-|-|-|
|Input resolution|256×256|384×384|
|Epochs|50|80|
|Augmentation bug|Present|Fixed|
|Inference TTA|None|Horizontal flip|
|Kaggle Dice|< 0.80|pending resubmission|
|Val Dice (UNet)|unknown|**0.8908**|
|Val Dice (ResNet34-UNet)|unknown|**0.9118**|

> \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*Takeaway:\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\* Dice went from < 0.80 to 0.89–0.91 without changing the model architecture — just by fixing a misplaced `return`, increasing resolution, and training longer. Auditing the data pipeline before tuning the model is almost always more efficient.

\---

## Architecture

### UNet

```
Input: (B, 3, 384, 384)
         │
   ┌─────▼─────┐
   │  Encoder  │  DoubleConv → MaxPool ×4
   │  64→128   │
   │  →256→512 │
   └─────┬─────┘
         │ skip connections (concat)
   ┌─────▼─────┐
   │Bottleneck │  DoubleConv  ch=1024
   └─────┬─────┘
         │
   ┌─────▼─────┐
   │  Decoder  │  ConvTranspose2d + concat skip + DoubleConv ×4
   │  512→256  │
   │  →128→64  │
   └─────┬─────┘
         │
   Conv1×1 → (B, 1, 384, 384)   raw logit
```

**\~31M parameters.** Skip connections concatenate encoder feature maps into the decoder at each scale, preserving spatial detail lost during downsampling.

### ResNet34-UNet

```
Input: (B, 3, 384, 384)
         │
┌────────▼────────────────────────────┐
│         ResNet-34 Encoder           │
│  stem  → s0 (B, 64,  /2)           │
│  layer1 → s1 (B, 64,  /4)  ×3 blocks│
│  layer2 → s2 (B, 128, /8)  ×4 blocks│
│  layer3 → s3 (B, 256, /16) ×6 blocks│
│  layer4 → s4 (B, 512, /32) ×3 blocks│
└────────┬────────────────────────────┘
         │  5-scale skip connections
┌────────▼────────────────────────────┐
│         UNet-style Decoder          │
│  dec3: upsample s4 + s3 → 256      │
│  dec2: upsample    + s2 → 128      │
│  dec1: upsample    + s1 → 64       │
│  dec0: upsample    + s0 → 32       │
│  ×2 upsample → Conv1×1             │
└────────┬────────────────────────────┘
         │
   (B, 1, 384, 384)   raw logit
```

**\~24M parameters.** Residual connections in the encoder improve gradient flow through depth. The stem-level skip (stride-2, before MaxPool) is an additional fifth connection not present in vanilla UNet, preserving early low-level features.

\---

## Implementation Details

### Data Preprocessing

* **Trimap binarisation**: label `1` → foreground (`1`); labels `2` \& `3` → background (`0`)
* **Resize**: 384×384 — bilinear for images, nearest-neighbour for masks
* **Normalisation**: ImageNet mean `\\\\\\\\\\\\\\\[0.485, 0.456, 0.406]` / std `\\\\\\\\\\\\\\\[0.229, 0.224, 0.225]`
* **Training augmentations** (applied identically to image and mask):
random horizontal flip · rotation ±15° · random resized crop · color jitter

### Loss Function

```
Loss = 0.5 × BCEWithLogitsLoss + 0.5 × SoftDiceLoss
```

BCE optimises per-pixel accuracy; Soft Dice directly optimises the evaluation metric and mitigates class imbalance by focusing on the overlap region.

### Training Configuration

|Setting|UNet|ResNet34-UNet|
|-|-|-|
|Optimiser|AdamW|AdamW|
|Learning rate|1e-3|5e-4|
|Weight decay|1e-4|1e-4|
|Scheduler|CosineAnnealingLR|CosineAnnealingLR|
|Batch size|8|4|
|Epochs|80|80|
|Gradient clip|max\_norm=1.0|max\_norm=1.0|

Best checkpoint selected by highest validation Dice Score.

\---

## Project Structure

```
├── dataset/
│   └── oxford-iiit-pet/          # not tracked by git — download separately
│       ├── images/
│       └── annotations/
│           ├── trimaps/
│           ├── trainval.txt
│           └── test.txt
├── images/                        # README figures
│   ├── model\\\\\\\\\\\\\\\_comparison.png
│   ├── unet\\\\\\\\\\\\\\\_training\\\\\\\\\\\\\\\_curves.png
│   └── resnet34\\\\\\\\\\\\\\\_unet\\\\\\\\\\\\\\\_training\\\\\\\\\\\\\\\_curves.png
├── src/
│   ├── models/
│   │   ├── \\\\\\\\\\\\\\\_\\\\\\\\\\\\\\\_init\\\\\\\\\\\\\\\_\\\\\\\\\\\\\\\_.py
│   │   ├── unet.py               # UNet from scratch
│   │   └── resnet34\\\\\\\\\\\\\\\_unet.py      # ResNet34 encoder + UNet decoder
│   ├── oxford\\\\\\\\\\\\\\\_pet.py             # dataset loader + augmentation
│   ├── utils.py                  # Dice score, BCE+Dice loss, visualisation
│   ├── train.py                  # training loop
│   ├── evaluate.py               # per-image Dice evaluation
│   └── inference.py              # Kaggle RLE submission (with TTA)
├── saved\\\\\\\\\\\\\\\_models/                 # trained checkpoints — not tracked by git
├── requirements.txt
└── README.md
```

\---

## Quick Start

### 1\. Install dependencies

```bash
git clone https://github.com/1eadbloom/YOUR\\\\\\\\\\\\\\\_REPO\\\\\\\\\\\\\\\_NAME.git
cd YOUR\\\\\\\\\\\\\\\_REPO\\\\\\\\\\\\\\\_NAME
pip install -r requirements.txt
```

### 2\. Download dataset

```bash
mkdir -p dataset/oxford-iiit-pet \\\\\\\\\\\\\\\&\\\\\\\\\\\\\\\& cd dataset/oxford-iiit-pet
wget https://www.robots.ox.ac.uk/\\\\\\\\\\\\\\\~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/\\\\\\\\\\\\\\\~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz \\\\\\\\\\\\\\\&\\\\\\\\\\\\\\\& tar -xf annotations.tar.gz \\\\\\\\\\\\\\\&\\\\\\\\\\\\\\\& cd ../..
```

### 3\. Train

```bash
# UNet
python src/train.py --model unet \\\\\\\\\\\\\\\\
    --epochs 80 --batch\\\\\\\\\\\\\\\_size 8 --lr 1e-3 \\\\\\\\\\\\\\\\
    --data\\\\\\\\\\\\\\\_root dataset/oxford-iiit-pet

# ResNet34 + UNet
python src/train.py --model resnet34\\\\\\\\\\\\\\\_unet \\\\\\\\\\\\\\\\
    --epochs 80 --batch\\\\\\\\\\\\\\\_size 4 --lr 5e-4 \\\\\\\\\\\\\\\\
    --data\\\\\\\\\\\\\\\_root dataset/oxford-iiit-pet
```

### 4\. Evaluate

```bash
python src/evaluate.py --model unet \\\\\\\\\\\\\\\\
    --checkpoint saved\\\\\\\\\\\\\\\_models/unet\\\\\\\\\\\\\\\_best.pth \\\\\\\\\\\\\\\\
    --data\\\\\\\\\\\\\\\_root dataset/oxford-iiit-pet

python src/evaluate.py --model resnet34\\\\\\\\\\\\\\\_unet \\\\\\\\\\\\\\\\
    --checkpoint saved\\\\\\\\\\\\\\\_models/resnet34\\\\\\\\\\\\\\\_unet\\\\\\\\\\\\\\\_best.pth \\\\\\\\\\\\\\\\
    --data\\\\\\\\\\\\\\\_root dataset/oxford-iiit-pet
```

### 5\. Generate Kaggle submission (with TTA)

```bash
python src/inference.py --model unet \\\\\\\\\\\\\\\\
    --checkpoint saved\\\\\\\\\\\\\\\_models/unet\\\\\\\\\\\\\\\_best.pth \\\\\\\\\\\\\\\\
    --data\\\\\\\\\\\\\\\_root dataset/oxford-iiit-pet \\\\\\\\\\\\\\\\
    --output submission\\\\\\\\\\\\\\\_unet.csv

python src/inference.py --model resnet34\\\\\\\\\\\\\\\_unet \\\\\\\\\\\\\\\\
    --checkpoint saved\\\\\\\\\\\\\\\_models/resnet34\\\\\\\\\\\\\\\_unet\\\\\\\\\\\\\\\_best.pth \\\\\\\\\\\\\\\\
    --data\\\\\\\\\\\\\\\_root dataset/oxford-iiit-pet \\\\\\\\\\\\\\\\
    --output submission\\\\\\\\\\\\\\\_resnet34\\\\\\\\\\\\\\\_unet.csv

# disable TTA
python src/inference.py --model unet \\\\\\\\\\\\\\\\
    --checkpoint saved\\\\\\\\\\\\\\\_models/unet\\\\\\\\\\\\\\\_best.pth --no\\\\\\\\\\\\\\\_tta
```

\---

## References

* Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
* Parkhi et al., *Cats and Dogs*, CVPR 2012 (Oxford-IIIT Pet Dataset)

\---

> \\\\\\\\\\\\\\\*AI assistance disclosure: Claude (Anthropic) assisted with original code debugging, partial hyperparameter configuration, and generating a summary of key implementation points for the technical report. Architecture decisions, training runs, and final results reflect the author's own work.\\\\\\\\\\\\\\\*

