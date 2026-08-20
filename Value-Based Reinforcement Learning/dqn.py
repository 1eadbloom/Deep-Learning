# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Student ID: 109205057

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse

gym.register_envs(ale_py)

# ──────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Network
# ──────────────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    """
    Design the architecture of your deep Q network.
    use_cnn=False → CartPole MLP
    use_cnn=True  → Atari CNN (Nature-DQN)
    """

    def __init__(self, num_actions, use_cnn=False):
        super(DQN, self).__init__()
        self.use_cnn = use_cnn

        ########## YOUR CODE HERE (5~10 lines) ##########
        if use_cnn:
            # Nature-DQN CNN  input: (B, 4, 84, 84)
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),   # → (B,32,20,20)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (B,64, 9, 9)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (B,64, 7, 7)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            # MLP for CartPole  input: (B, 4)
            self.network = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.use_cnn:
            return self.network(x / 255.0)
        return self.network(x.float())


# ──────────────────────────────────────────────────────────────────────────────
# Atari Pre-processor
# ──────────────────────────────────────────────────────────────────────────────

class AtariPreprocessor:
    """Grayscale + resize to 84×84 + 4-frame stack."""

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) \
               if (len(obs.shape) == 3 and obs.shape[2] == 3) else obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(self.preprocess(obs))
        return np.stack(self.frames, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Prioritised Experience Replay  (Task 3)
# ──────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Prioritized replay (Schaul et al., 2016).
    https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta        = beta
        self.buffer      = []
        self.priorities  = np.zeros((capacity,), dtype=np.float32)
        self.pos         = 0
        self.max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########
        priority = (abs(float(error)) + 1e-6) ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        n      = len(self.buffer)
        probs  = self.priorities[:n] / self.priorities[:n].sum()
        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()            # normalise
        ########## END OF YOUR CODE (for Task 3) ##########
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, err in zip(indices, errors):
            p = (abs(float(err)) + 1e-6) ** self.alpha
            self.priorities[idx]  = p
            self.max_priority     = max(self.max_priority, p)
        ########## END OF YOUR CODE (for Task 3) ##########


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name  = env_name
        self.is_atari  = "Pong" in env_name or "ALE" in env_name

        self.env      = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor() if self.is_atari else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # ── Networks ──────────────────────────────────────────────────────────
        self.q_net      = DQN(self.num_actions, use_cnn=self.is_atari).to(self.device)
        self.target_net = DQN(self.num_actions, use_cnn=self.is_atari).to(self.device)

        if getattr(args, "resume_path", None):
            ckpt = torch.load(args.resume_path, map_location=self.device,
                              weights_only=True)
            self.q_net.load_state_dict(ckpt)
            self.target_net.load_state_dict(ckpt)
            print(f"[Resume] Loaded weights from {args.resume_path}")
        else:
            self.q_net.apply(init_weights)
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # ── Hyper-parameters ──────────────────────────────────────────────────
        self.batch_size              = args.batch_size
        self.gamma                   = args.discount_factor
        self.epsilon_min             = args.epsilon_min
        self.epsilon_decay           = args.epsilon_decay
        self.max_episode_steps       = args.max_episode_steps
        self.replay_start_size       = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step          = args.train_per_step
        self.save_dir                = args.save_dir
        self.student_id              = getattr(args, "student_id", "109205057")
        os.makedirs(self.save_dir, exist_ok=True)

        # Resumable counters
        # getattr fallback only works when attribute is missing entirely;
        # when --initial-epsilon is not passed it is None (not missing),
        # so we need an explicit None check here.
        _ie = getattr(args, "initial_epsilon", None)
        self.epsilon     = _ie if _ie is not None else args.epsilon_start
        _ib = getattr(args, "initial_best_reward", None)
        self.best_reward = _ib if _ib is not None else (-21.0 if self.is_atari else 0.0)
        self.env_count   = getattr(args, "initial_env_steps",   0)
        self.train_count = getattr(args, "initial_train_count", 0)

        # ── Task-3 enhancements ───────────────────────────────────────────────
        self.use_double_dqn  = args.use_double_dqn
        self.use_per         = args.use_per
        self.n_step          = args.n_step
        self.per_beta_start  = args.per_beta_start
        self.per_beta_frames = args.per_beta_frames

        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                args.memory_size,
                alpha=args.per_alpha,
                beta=self.per_beta_start,
            )
        else:
            self.memory = deque(maxlen=args.memory_size)

        # n-step transition buffer
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Snapshot milestones for Task-3
        self.snapshot_steps  = sorted(set(args.snapshot_steps)) \
                               if args.snapshot_steps else []
        self.saved_snapshots = set()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_state(self, obs, reset=False):
        if self.is_atari:
            return self.preprocessor.reset(obs) if reset \
                   else self.preprocessor.step(obs)
        return np.array(obs, dtype=np.float32)

    def _anneal_per_beta(self):
        if not self.use_per:
            return
        progress = min(1.0, self.env_count / max(1, self.per_beta_frames))
        self.memory.beta = self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def _store_transition(self, state, action, reward, next_state, done):
        """Handle 1-step or n-step buffering before pushing to replay."""
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            if len(self.n_step_buffer) < self.n_step:
                return          # buffer not full yet
            R = sum((self.gamma ** i) * r
                    for i, (_, _, r, _, _) in enumerate(self.n_step_buffer))
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            s_n, d_n = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            self._push((s0, a0, R, s_n, d_n))
            if done:
                self._flush_n_step()
        else:
            self._push((state, action, reward, next_state, done))

    def _flush_n_step(self):
        """Drain remaining items in n-step buffer at episode end."""
        while self.n_step_buffer:
            R = sum((self.gamma ** i) * r
                    for i, (_, _, r, _, _) in enumerate(self.n_step_buffer))
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            s_n, d_n = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            self._push((s0, a0, R, s_n, d_n))
            self.n_step_buffer.popleft()

    def _push(self, transition):
        if self.use_per:
            self.memory.add(transition, self.memory.max_priority)
        else:
            self.memory.append(transition)

    def _maybe_save_snapshot(self):
        for step in self.snapshot_steps:
            if step not in self.saved_snapshots and self.env_count >= step:
                fname = f"LAB5_{self.student_id}_task3_{step}.pt"
                path  = os.path.join(self.save_dir, fname)
                torch.save(self.q_net.state_dict(), path)
                self.saved_snapshots.add(step)
                print(f"[Snapshot] Saved {path}  (env_step={self.env_count})")

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        t = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(t).argmax().item()

    # ── Training loop ─────────────────────────────────────────────────────────

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _      = self.env.reset()
            state       = self._get_state(obs, reset=True)
            done        = False
            total_reward = 0
            step_count   = 0
            self.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done        = terminated or truncated
                next_state  = self._get_state(next_obs)

                self._store_transition(state, action, reward, next_state, done)

                for _ in range(self.train_per_step):
                    self.train()

                state         = next_state
                total_reward += reward
                self.env_count += 1
                step_count     += 1
                self._maybe_save_snapshot()

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep:{ep} Step:{step_count} "
                          f"SC:{self.env_count} UC:{self.train_count} "
                          f"Eps:{self.epsilon:.4f}")
                    wandb.log({
                        "Env Step Count": self.env_count,
                        "Update Count":   self.train_count,
                        "Epsilon":        self.epsilon,
                    })

            if done:
                self._flush_n_step()

            print(f"[Ep {ep}] Reward:{total_reward}  "
                  f"SC:{self.env_count} UC:{self.train_count} "
                  f"Eps:{self.epsilon:.4f}")
            wandb.log({
                "Episode":        ep,
                "Total Reward":   total_reward,
                "Env Step Count": self.env_count,
                "Update Count":   self.train_count,
                "Epsilon":        self.epsilon,
            })

            # Periodic checkpoint every 100 episodes
            if ep % 100 == 0:
                path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), path)
                print(f"[Checkpoint] {path}")

            # Eval + best-model every 20 episodes
            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), best)
                    print(f"[Best] reward={eval_reward:.2f}  → {best}")
                print(f"[TrueEval] Ep:{ep} Eval:{eval_reward:.2f} "
                      f"SC:{self.env_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count":   self.train_count,
                    "Eval Reward":    eval_reward,
                })

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, episodes=5):
        rewards = []
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state  = self._get_state(obs, reset=True)
            done   = False
            total  = 0
            while not done:
                t = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(t).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done  = terminated or truncated
                total += reward
                state  = self._get_state(next_obs)
            rewards.append(total)
        return float(np.mean(rewards))

    # ── Training step ─────────────────────────────────────────────────────────

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
        self._anneal_per_beta()

        # ---- Sample mini-batch -----------------------------------------------
        ########## YOUR CODE HERE (<5 lines) ##########
        if self.use_per:
            batch, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32,
                                   device=self.device)
        else:
            batch   = random.sample(self.memory, self.batch_size)
            indices = None
            weights = None
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        states      = torch.from_numpy(np.array(states,      dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.int64,   device=self.device)
        rewards_t   = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t     = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ---- Loss and gradient update ----------------------------------------
        ########## YOUR CODE HERE (~10 lines) ##########
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: online net selects action; target net evaluates
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]

            gamma_n = self.gamma ** self.n_step
            target  = rewards_t + (1.0 - dones_t) * gamma_n * next_q

        td_errors = target - q_values

        if self.use_per and weights is not None:
            # Weighted Huber loss (more stable than MSE)
            loss = (weights * nn.functional.huber_loss(
                        q_values, target, reduction='none')).mean()
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        else:
            loss = nn.functional.huber_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        # Sync target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] "
                  f"Loss:{loss.item():.4f}  "
                  f"Q_mean:{q_values.mean().item():.3f}  "
                  f"Q_std:{q_values.std().item():.3f}")
            wandb.log({
                "Loss":           loss.item(),
                "Q Mean":         q_values.mean().item(),
                "Env Step Count": self.env_count,
                "Update Count":   self.train_count,
            })


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Lab5 – DQN Training")

    # ── Identity / IO ─────────────────────────────────────────────────────────
    parser.add_argument("--env",              type=str,   default="CartPole-v1")
    parser.add_argument("--episodes",         type=int,   default=1000)
    parser.add_argument("--save-dir",         type=str,   default="./results")
    parser.add_argument("--student-id",       type=str,   default="109205057")
    parser.add_argument("--wandb-run-name",   type=str,   default="dqn-run")
    parser.add_argument("--wandb-project",    type=str,   default="DLP-Lab5-DQN")
    parser.add_argument("--wandb-mode",       type=str,   default="online",
                        help="wandb mode: online / offline / disabled")

    # ── Core hyper-parameters ─────────────────────────────────────────────────
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--memory-size",      type=int,   default=100_000)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--discount-factor",  type=float, default=0.99)
    parser.add_argument("--epsilon-start",    type=float, default=1.0)
    parser.add_argument("--epsilon-decay",    type=float, default=0.999_999)
    parser.add_argument("--epsilon-min",      type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1_000)
    parser.add_argument("--replay-start-size",        type=int, default=50_000)
    parser.add_argument("--max-episode-steps",        type=int, default=10_000)
    parser.add_argument("--train-per-step",   type=int,   default=4)

    # ── Task-3 enhancements ───────────────────────────────────────────────────
    parser.add_argument("--use-double-dqn",  action="store_true")
    parser.add_argument("--use-per",         action="store_true")
    parser.add_argument("--n-step",          type=int,   default=1)
    parser.add_argument("--per-alpha",       type=float, default=0.6)
    parser.add_argument("--per-beta-start",  type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int,   default=1_000_000)
    parser.add_argument("--snapshot-steps",  type=int,   nargs="*", default=[])

    # ── Resume ────────────────────────────────────────────────────────────────
    parser.add_argument("--resume-path",          type=str,   default=None)
    parser.add_argument("--initial-env-steps",    type=int,   default=0)
    parser.add_argument("--initial-train-count",  type=int,   default=0)
    parser.add_argument("--initial-epsilon",      type=float, default=None)
    parser.add_argument("--initial-best-reward",  type=float, default=None)

    args = parser.parse_args()

    # ── Environment-specific tweaks ───────────────────────────────────────────
    if "CartPole" in args.env:
        # Fast defaults for CartPole
        if args.replay_start_size == 50_000: args.replay_start_size = 1_000
        if args.epsilon_decay == 0.999_999:  args.epsilon_decay      = 0.9995
        if args.target_update_frequency == 1_000: args.target_update_frequency = 500
        if args.lr == 1e-4:                  args.lr                 = 5e-4
        if args.train_per_step == 4:         args.train_per_step     = 1
    elif "Pong" in args.env or "ALE" in args.env:
        args.env = "ALE/Pong-v5"
        if args.epsilon_decay == 0.999_999:  args.epsilon_decay = 0.999_985
        # Auto-set snapshot milestones for Task 3
        if not args.snapshot_steps and \
           (args.use_per or args.use_double_dqn or args.n_step > 1):
            args.snapshot_steps = [600_000, 1_000_000, 1_500_000,
                                   2_000_000, 2_500_000]

    # When resuming, warn if replay_start_size is too high
    # (buffer is empty after restart, so a large value wastes time)
    # Fix: use --replay-start-size 5000 in your resume .bat explicitly.
    if args.resume_path and args.replay_start_size >= 10_000:
        print(f"[Resume] Warning: --replay-start-size={args.replay_start_size} is large.")
        print(f"[Resume] The replay buffer is empty after restart.")
        print(f"[Resume] Recommend: add --replay-start-size 5000 to your resume command.")

    # ── WandB + Training ──────────────────────────────────────────────────────
    wandb.init(project=args.wandb_project,
               name=args.wandb_run_name,
               save_code=True,
               config=vars(args),
               anonymous='allow',
               mode=getattr(args, 'wandb_mode', 'online'))
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run(episodes=args.episodes)
