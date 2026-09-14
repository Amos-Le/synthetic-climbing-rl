import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from envs.climb_env import ClimbEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class CurriculumCallback(BaseCallback):
    """Ramps env difficulty from 0.0 (easy, near goal) to 1.0 (true start)
    linearly over the course of training.

    difficulty_start_frac / difficulty_end_frac control WHEN the ramp
    happens relative to total training progress — e.g. spend the first
    20% of training purely at difficulty 0.0 before starting to ramp,
    so the policy has a solid easy-mode baseline before the task gets
    harder.
    """

    def __init__(self, total_timesteps, difficulty_start_frac=0.1,
                 difficulty_end_frac=0.9, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.difficulty_start_frac = difficulty_start_frac
        self.difficulty_end_frac = difficulty_end_frac

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        if progress < self.difficulty_start_frac:
            difficulty = 0.0
        elif progress > self.difficulty_end_frac:
            difficulty = 1.0
        else:
            span = self.difficulty_end_frac - self.difficulty_start_frac
            difficulty = (progress - self.difficulty_start_frac) / span

        # env is wrapped in a DummyVecEnv by SB3 — reach through to the
        # underlying ClimbEnv instance(s) to call set_difficulty()
        for e in self.training_env.envs:
            e.unwrapped.set_difficulty(difficulty)

        if self.verbose and self.num_timesteps % 10000 == 0:
            print(f"[curriculum] timestep={self.num_timesteps} difficulty={difficulty:.3f}")

        return True


TOTAL_TIMESTEPS = 1000000

env = ClimbEnv(difficulty=0.0)

model = PPO("MlpPolicy", env, verbose=1)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="outputs/trained_models/checkpoints",
    name_prefix="climb_ppo"
)

curriculum_callback = CurriculumCallback(
    total_timesteps=TOTAL_TIMESTEPS,
    difficulty_start_frac=0.1,
    difficulty_end_frac=0.9,
    verbose=1,
)

print("=== Starting extended training run (1,000,000 timesteps, best config: fixed reward, difficulty=0.0, no curriculum) ===")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback],  # curriculum stays disabled — this config performed best
)

model.save("outputs/trained_models/climb_ppo_v7")
print("Saved final model to outputs/trained_models/climb_ppo_v7")

env.close()
