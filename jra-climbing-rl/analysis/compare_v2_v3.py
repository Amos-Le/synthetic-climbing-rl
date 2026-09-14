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

print("Evaluating climb_ppo_v2 (old reward, 300K)...")
summary_v2 = report("v2", run_eval("outputs/trained_models/climb_ppo_v2", difficulty=0.0))

print("\nEvaluating climb_ppo_v3 (new reward + auto curriculum, 300K)...")
summary_v3 = report("v3", run_eval("outputs/trained_models/climb_ppo_v3", difficulty=0.0))

print("\n" + "=" * 55)
print("COMPARISON — evaluated at difficulty=0.0 for both (fair comparison)")
print("=" * 55)
print(f"{'Metric':<28} {'v2':<14} {'v3':<14}")
print(f"{'Avg episode length':<28} {summary_v2['avg_steps']:<14.1f} {summary_v3['avg_steps']:<14.1f}")
print(f"{'Pose error (lower=better)':<28} {summary_v2['pose_error_mean']:<14.3f} {summary_v3['pose_error_mean']:<14.3f}")
print(f"{'Smoothness (lower=better)':<28} {summary_v2['smoothness_mean']:<14.4f} {summary_v3['smoothness_mean']:<14.4f}")
print(f"{'Success rate':<28} {summary_v2['success_rate']:<14.1%} {summary_v3['success_rate']:<14.1%}")