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

print("Evaluating climb_ppo_v4 (fixed reward, no curriculum) at difficulty=0.0...")
summary_v4 = report("v4 @ 0.0", run_eval("outputs/trained_models/climb_ppo_v4", difficulty=0.0))

print("\nEvaluating climb_ppo_v5 (fixed reward + curriculum) at difficulty=0.0 (easy, same as v4)...")
summary_v5_easy = report("v5 @ 0.0 (easy)", run_eval("outputs/trained_models/climb_ppo_v5", difficulty=0.0))

print("\nEvaluating climb_ppo_v5 at difficulty=1.0 (its full trained task)...")
summary_v5_hard = report("v5 @ 1.0 (full task)", run_eval("outputs/trained_models/climb_ppo_v5", difficulty=1.0))

print("\nEvaluating climb_ppo_v3 (old reward + curriculum) at difficulty=1.0, for reference...")
summary_v3_hard = report("v3 @ 1.0 (old reward, for reference)", run_eval("outputs/trained_models/climb_ppo_v3", difficulty=1.0))

print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)
print(f"{'Model':<32} {'Len':<8} {'PoseErr':<10} {'Smooth':<10} {'Success'}")
def row(label, s):
    print(f"{label:<32} {s['avg_steps']:<8.1f} {s['pose_error_mean']:<10.2f} {s['smoothness_mean']:<10.4f} {s['success_rate']:.1%}")
row("v4 (fixed reward) @ 0.0", summary_v4)
row("v5 (reward+curriculum) @ 0.0", summary_v5_easy)
row("v5 (reward+curriculum) @ 1.0", summary_v5_hard)
row("v3 (old reward+curriculum) @ 1.0", summary_v3_hard)