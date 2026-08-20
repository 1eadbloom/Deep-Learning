"""
evaluate_model.py – Official Lab5 evaluation (20 episodes, seeds 0-19)

Usage:
  python evaluate_model.py --task 1 --model-path ./results/task1/best_model.pt
  python evaluate_model.py --task 2 --model-path ./results/task2/best_model.pt
  python evaluate_model.py --task 3 --model-path ./results/task3/LAB5_109205057_task3_2500000.pt
"""

import argparse, os, random
import gymnasium as gym
import numpy as np
import torch
import ale_py

from dqn import DQN, AtariPreprocessor

gym.register_envs(ale_py)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.task in (2, 3):
        env_name = "ALE/Pong-v5"
        use_cnn  = True
    else:
        env_name = "CartPole-v1"
        use_cnn  = False

    env          = gym.make(env_name, render_mode="rgb_array")
    num_actions  = env.action_space.n
    preprocessor = AtariPreprocessor() if use_cnn else None

    model = DQN(num_actions, use_cnn=use_cnn).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {args.model_path}")

    rewards = []
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env.action_space.seed(seed); env.observation_space.seed(seed)

        obs, _ = env.reset(seed=seed)
        state  = preprocessor.reset(obs) if preprocessor \
                 else np.array(obs, dtype=np.float32)

        done, total = False, 0
        while not done:
            t = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(t).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            total += reward
            state  = preprocessor.step(next_obs) if preprocessor \
                     else np.array(next_obs, dtype=np.float32)

        rewards.append(total)
        print(f"  seed={seed:>2}  reward={total:>6.1f}")

    mean, std = float(np.mean(rewards)), float(np.std(rewards))
    print("=" * 50)
    print(f"Model : {args.model_path}")
    print(f"Mean  : {mean:.2f}  ±{std:.2f}  ({args.episodes} episodes)")

    # Grading estimate
    if args.task == 1:
        score = min(mean, 480) / 480 * 15
        print(f"[Task 1] Grading ≈ {score:.2f} / 15")
    elif args.task == 2:
        score = (min(mean, 19) + 21) / 40 * 20
        print(f"[Task 2] Grading ≈ {score:.2f} / 20")
    print("=" * 50)

    env.close()
    return mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)
