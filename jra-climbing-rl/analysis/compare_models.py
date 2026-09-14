import numpy as np
from stable_baselines3 import PPO

from envs.climb_env import ClimbEnv
from evaluation.evaluate_policy import rollout_episode
from evaluation.metrics import episode_pose_error, episode_smoothness, success_rate, summarize

def run_eval(model_path, n_episodes=10, difficulty=0.0):
    env = ClimbEnv(difficulty=difficulty)
    model = PPO.load(model_path)

    episodes = []
    for ep in range(n_episodes):
        result = rollout_episode(env, model)
        episodes.append(result)

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

    return {
        "avg_steps": avg_steps,
        "pose_error_mean": float(np.nanmean(pose_errors)),
        "smoothness_mean": float(np.nanmean(smoothness)),
        "success_rate": succ_rate,
    }

print("Evaluating climb_ppo_v1 (20K timesteps)...")
episodes_v1 = run_eval("outputs/trained_models/climb_ppo_v1", n_episodes=10, difficulty=0.0)
summary_v1 = report("v1 (20K timesteps)", episodes_v1)

print("\nEvaluating climb_ppo_v2 (300K timesteps)...")
episodes_v2 = run_eval("outputs/trained_models/climb_ppo_v2", n_episodes=10, difficulty=0.0)
summary_v2 = report("v2 (300K timesteps)", episodes_v2)

print("\n" + "=" * 55)
print("SIDE-BY-SIDE COMPARISON (10 episodes each)")
print("=" * 55)
print(f"{'Metric':<28} {'v1 (20K)':<14} {'v2 (300K)':<14}")
print(f"{'Avg episode length':<28} {summary_v1['avg_steps']:<14.1f} {summary_v2['avg_steps']:<14.1f}")
print(f"{'Pose error (lower=better)':<28} {summary_v1['pose_error_mean']:<14.3f} {summary_v2['pose_error_mean']:<14.3f}")
print(f"{'Smoothness (lower=better)':<28} {summary_v1['smoothness_mean']:<14.4f} {summary_v2['smoothness_mean']:<14.4f}")
print(f"{'Success rate':<28} {summary_v1['success_rate']:<14.1%} {summary_v2['success_rate']:<14.1%}")