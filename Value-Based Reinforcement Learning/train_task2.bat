@echo off
REM ── Task 2: Vanilla DQN on ALE/Pong-v5 ──────────────────────────────────────
REM Expected runtime: 15-25 hours on RTX 5080 Laptop
REM Tip: use train_task2_resume.bat if training is interrupted
python dqn.py ^
  --env ALE/Pong-v5 ^
  --episodes 5000 ^
  --save-dir ./results/task2 ^
  --student-id 109205057 ^
  --wandb-project DLP-Lab5-DQN-Pong ^
  --wandb-run-name task2-pong-vanilla ^
  --batch-size 32 ^
  --memory-size 100000 ^
  --lr 0.0001 ^
  --discount-factor 0.99 ^
  --epsilon-start 1.0 ^
  --epsilon-decay 0.999985 ^
  --epsilon-min 0.05 ^
  --target-update-frequency 1000 ^
  --replay-start-size 50000 ^
  --max-episode-steps 10000 ^
  --train-per-step 4
