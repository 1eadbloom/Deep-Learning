@echo off
REM ── Official evaluation (seeds 0-19) for all three tasks ──────────────────
SET ID=109205057

echo.
echo ========== TASK 1  CartPole ==========
python evaluate_model.py ^
  --task 1 ^
  --model-path ./results/task1/best_model.pt ^
  --episodes 20

echo.
echo ========== TASK 2  Pong (vanilla) ==========
python evaluate_model.py ^
  --task 2 ^
  --model-path ./results/task2/best_model.pt ^
  --episodes 20

echo.
echo ========== TASK 3  Pong (enhanced) – all milestones ==========
FOR %%S IN (600000 1000000 1500000 2000000 2500000) DO (
  SET SPATH=./results/task3/LAB5_%ID%_task3_%%S.pt
  IF EXIST !SPATH! (
    echo.
    echo --- Milestone %%S ---
    python evaluate_model.py --task 3 --model-path !SPATH! --episodes 20
  ) ELSE (
    echo [SKIP] !SPATH! not found
  )
)

echo.
echo ========== TASK 3  best model ==========
python evaluate_model.py ^
  --task 3 ^
  --model-path ./results/task3/best_model.pt ^
  --episodes 20
