import numpy as np
from stable_baselines3 import PPO

from envs.climb_env import ClimbEnv
from evaluation.evaluate_policy import rollout_episode
from evaluation.metrics import episode_pose_error, episode_smoothness, success_rate, summarize

def run_eval(model_path, n_episodes=10, difficulty=0.0):
    env = ClimbEnv(difficulty=difficulty)
    model = PPO.load(model_path)
    episodes = [rollout_episode(env, model) for _ in range(n_episodes)]
    env.close()
    return episodes

def report(label, episodes):
    pose_errors = [episode_pose_error(ep) for ep in episodes]
    smoothness = [episode_smoothness(ep) for ep in episodes]
    succ_rate = success_rate(episodes)
    avg_steps = np.mean([ep["n_steps"] for ep in episodes])
    print(f"\n--- {label} ---")
    print(f"Average episode length: {avg_steps:.1f} steps")
    print(summarize("Pose imitation error", pose_errors))
    print(summarize("Action smoothness", smoothness))
    print(f"Success rate (no fall): {succ_rate:.1%}")
    return {"avg_steps": avg_steps, "pose_error_mean": float(np.nanmean(pose_errors)),
            "smoothness_mean": float(np.nanmean(smoothness)), "success_rate": succ_rate}

print("Evaluating climb_ppo_v2 at ITS home condition (difficulty=0.0)...")
summary_v2 = report("v2 @ difficulty=0.0 (trained condition)", run_eval("outputs/trained_models/climb_ppo_v2", difficulty=0.0))

print("\nEvaluating climb_ppo_v3 at ITS home condition (difficulty=1.0)...")
summary_v3 = report("v3 @ difficulty=1.0 (trained condition)", run_eval("outputs/trained_models/climb_ppo_v3", difficulty=1.0))

print("\n" + "=" * 60)
print("MATCHED COMPARISON — each model evaluated at its own trained difficulty")
print("=" * 60)
print(f"{'Metric':<28} {'v2 @ 0.0':<16} {'v3 @ 1.0':<16}")
print(f"{'Avg episode length':<28} {summary_v2['avg_steps']:<16.1f} {summary_v3['avg_steps']:<16.1f}")
print(f"{'Pose error (lower=better)':<28} {summary_v2['pose_error_mean']:<16.3f} {summary_v3['pose_error_mean']:<16.3f}")
print(f"{'Smoothness (lower=better)':<28} {summary_v2['smoothness_mean']:<16.4f} {summary_v3['smoothness_mean']:<16.4f}")
print(f"{'Success rate':<28} {summary_v2['success_rate']:<16.1%} {summary_v3['success_rate']:<16.1%}")