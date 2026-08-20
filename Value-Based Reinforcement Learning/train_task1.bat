@echo off
REM ── Task 1: Vanilla DQN on CartPole-v1 ──────────────────────────────────────
REM Expected runtime: ~5-10 minutes on CPU/GPU
python dqn.py ^
  --env CartPole-v1 ^
  --episodes 800 ^
  --save-dir ./results/task1 ^
  --student-id 109205057 ^
  --wandb-project DLP-Lab5-DQN ^
  --wandb-run-name task1-cartpole ^
  --batch-size 64 ^
  --memory-size 50000 ^
  --lr 0.0005 ^
  --discount-factor 0.99 ^
  --epsilon-start 1.0 ^
  --epsilon-decay 0.9995 ^
  --epsilon-min 0.01 ^
  --target-update-frequency 500 ^
  --replay-start-size 1000 ^
  --max-episode-steps 500 ^
  --train-per-step 1
