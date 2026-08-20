# Value-Based Reinforcement Learning

> **Deep Learning Lab5** — NYCU, Spring 2026  
> Implementation of Deep Q-Network (DQN) and its enhancements for solving
classic control and Atari visual reinforcement learning tasks.
---
## Overview

| Task | Environment | Method | Result |
|------|-------------|--------|--------|
| 1 | CartPole-v1 | Vanilla DQN (MLP) | Avg ≈ 480 |
| 2 | ALE/Pong-v5 | Vanilla DQN (CNN) | Avg ≈ 14.2 |
| 3 | ALE/Pong-v5 | Double DQN + PER + Multi-step | Best ≈ +1.05 |

---

## Methods

### Task 1 – Vanilla DQN on CartPole
- 3-layer MLP (128 → 128 → actions)
- Epsilon-greedy exploration with exponential decay
- Experience replay with uniform sampling
- Hard target network update every 500 steps

### Task 2 – Vanilla DQN on Atari Pong
- Nature-DQN CNN: Conv(8x4) → Conv(4x2) → Conv(3x1) → FC(512) → actions
- 4-frame grayscale stack, 84x84 resize
- 100k replay buffer

### Task 3 – Enhanced DQN
- **Double DQN**: decouple action selection from evaluation
- **Prioritized Experience Replay (PER)**: alpha=0.6, beta annealed 0.4→1.0
- **Multi-step Return (n=3)**: 3-step discounted reward accumulation

---

## Project Structure

```
Lab5_VSCode/
├── dqn.py                   # Main training script (Tasks 1-3)
├── evaluate_model.py        # Official 20-episode evaluation (seeds 0-19)
├── requirements.txt
├── train_task1.bat
├── train_task2.bat
├── train_task2_resume.bat
├── train_task3.bat
├── train_task3_resume.bat
├── eval_task1.bat
├── eval_task2.bat
├── eval_task3_all.bat
└── .gitignore
```

> Model weights (.pt) and training outputs (results/, wandb/) are excluded
> from this repo due to file size limits.

---

## Setup

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA (RTX 5080 / SM 12.0 / cu128)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Train
.\train_task1.bat   # ~10 min
.\train_task2.bat   # ~20 hrs on RTX 5080 Laptop
.\train_task3.bat   # ~15 hrs on RTX 5080 Laptop

# 4. Evaluate
python evaluate_model.py --task 1 --model-path ./results/task1/best_model.pt
python evaluate_model.py --task 2 --model-path ./results/task2/best_model.pt
python evaluate_model.py --task 3 --model-path ./results/task3/LAB5_109205057_task3_2500000.pt
```

---

## Key Notes

- **Resume**: buffer is NOT saved between runs; use `--replay-start-size 5000` on resume
- **Task 3 snapshots**: auto-saved at 600k/1M/1.5M/2M/2.5M env steps
- **Windows**: run .bat from CMD to avoid encoding issues; use detached window to prevent forrtl crashes

---

## Training Curves (WandB)
- Task 2: https://wandb.ai/1eadbloom-none/DLP-Lab5-DQN-Pong
- Task 3: https://wandb.ai/1eadbloom-none/DLP-Lab5-DQN-Pong-Enhanced

---

## References
1. Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
2. van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.
3. Schaul et al. (2016). Prioritized Experience Replay. ICLR.
4. Hessel et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. AAAI.
