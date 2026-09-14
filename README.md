# Synthetic Human Climbing Motion via Reinforcement Learning

Generating synthetic stair-climbing motion for a simulated humanoid using reinforcement learning, guided by real human motion-capture data. Built as part of the International Junior Research Associate (IJRA) Programme, University of Sussex.

## Overview

This project adapts a hierarchical reinforcement learning approach (originally developed for human-object interaction, Chao et al. 2021) to generate realistic, human-like climbing motion. A physics-simulated humanoid learns to climb a single step through trial and error (PPO), guided by a reward that combines goal-reaching with similarity to real CMU Motion Capture data.

## Project structure

```
envs/
  climb_env.py          # Custom Gymnasium environment wrapping the MuJoCo simulation
  reward.py             # Reward function: goal-distance + motion-imitation similarity
  humanoid_climb.xml    # MuJoCo world: humanoid + staircase

controllers/
  train.py              # PPO training entrypoint (Stable-Baselines3)

evaluation/
  evaluate_policy.py    # Rolls out a trained policy, records trajectories
  metrics.py            # Pose-similarity error, smoothness, success rate

build_reference_traj.py # Converts raw CMU MoCap data into a usable reference trajectory
```

Large generated artifacts (trained model checkpoints, raw/processed motion-capture data) are excluded from this repo via `.gitignore` — see below for how to regenerate them.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

**Convert reference motion-capture data:**
```bash
python build_reference_traj.py
```

**Train a policy:**
```bash
python controllers/train.py
```

**Evaluate a trained policy:**
```bash
python -m evaluation.evaluate_policy
```

## Method summary

1. **Simulate** — a MuJoCo humanoid and static staircase
2. **Sense & Act** — the agent observes joint angles/velocities, root state, and a goal signal, and outputs joint torques
3. **Reward** — combines proximity to the goal with similarity to real human climbing motion (CMU MoCap)
4. **Learn** — PPO (via Stable-Baselines3) updates the policy over many training episodes

## Key finding: reward hacking and its correction

An early version of the reward function was found to reward the agent for lunging toward the goal rather than climbing in a balanced, human-like way — a classic reward-hacking failure mode, diagnosed through controlled before/after evaluation rather than training reward alone. This was corrected with a survival bonus and fall penalty, verified via an isolated comparison to genuinely improve balance, human-likeness, and motion smoothness simultaneously. Full details are in the accompanying paper.

## Data source

Reference motion data: [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu) (subject 13, "climb 3 steps").

## Author

Amos — International Junior Research Associate Programme, University of Sussex, supervised by Temitayo Olugbade.
