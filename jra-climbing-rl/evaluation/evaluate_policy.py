"""
evaluate_policy.py

Rolls out a trained PPO checkpoint deterministically for N episodes,
saves per-episode rollouts (for later synthetic-data export / inspection),
and prints summary evaluation numbers using metrics.py.

Usage:
    python evaluate_policy.py --model outputs/trained_models/climb_ppo_v2 --episodes 20
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO

from envs.climb_env import ClimbEnv
from evaluation.metrics import (
    episode_pose_error,
    episode_smoothness,
    success_rate,
    summarize,
)


def rollout_episode(env, model, max_steps=300):
    """Run one deterministic episode and record everything metrics.py needs.

    Returns:
        dict with obs trajectory, actions, rewards, root heights, and
        whether the episode ended in termination (fall) vs truncation
        (ran out of time / reached the cap).
    """
    obs, info = env.reset()
    obs_traj = [obs.copy()]
    actions = []
    rewards = []
    root_heights = [env.data.qpos[2]]
    goal_rewards = []
    imitation_rewards = []

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_traj.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)
        root_heights.append(env.data.qpos[2])
        goal_rewards.append(info["goal_reward"])
        imitation_rewards.append(info["imitation_reward"])

        step += 1

    return {
        "obs": np.array(obs_traj),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "root_heights": np.array(root_heights),
        "goal_rewards": np.array(goal_rewards),
        "imitation_rewards": np.array(imitation_rewards),
        "terminated": terminated,   # True = fell
        "truncated": truncated,     # True = survived to episode cap
        "n_steps": step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/trained_models/climb_ppo_v2")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=float, default=1.0,
                         help="1.0 = full task (true start), 0.0 = easy start near goal")
    parser.add_argument("--save-dir", type=str, default="outputs/eval_rollouts")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = ClimbEnv(difficulty=args.difficulty)
    model = PPO.load(args.model)

    episodes = []
    for ep in range(args.episodes):
        result = rollout_episode(env, model)
        episodes.append(result)

        np.savez(
            os.path.join(args.save_dir, f"episode_{ep:03d}.npz"),
            obs=result["obs"],
            actions=result["actions"],
            rewards=result["rewards"],
            root_heights=result["root_heights"],
            terminated=result["terminated"],
            truncated=result["truncated"],
            n_steps=result["n_steps"],
        )

        print(
            f"Episode {ep:02d}: steps={result['n_steps']:3d} | "
            f"total_reward={result['rewards'].sum():7.2f} | "
            f"fell={result['terminated']} | "
            f"final_root_height={result['root_heights'][-1]:.3f}"
        )

    env.close()

    # --- Summary metrics across all episodes ---
    pose_errors = [episode_pose_error(ep) for ep in episodes]
    smoothness = [episode_smoothness(ep) for ep in episodes]
    succ_rate = success_rate(episodes)

    print("\n=== Evaluation summary over {} episodes ===".format(args.episodes))
    print(summarize("Mean imitation pose error (z-score units)", pose_errors))
    print(summarize("Action smoothness (mean |delta action|)", smoothness))
    print(f"Success rate (survived to episode cap without falling): {succ_rate:.1%}")

    np.savez(
        os.path.join(args.save_dir, "summary.npz"),
        pose_errors=pose_errors,
        smoothness=smoothness,
        success_rate=succ_rate,
    )
    print(f"\nSaved per-episode rollouts and summary to {args.save_dir}/")


if __name__ == "__main__":
    main()
