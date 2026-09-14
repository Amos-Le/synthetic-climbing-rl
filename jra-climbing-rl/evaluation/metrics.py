"""
metrics.py

Evaluation metrics for a rolled-out episode dict, as produced by
evaluate_policy.rollout_episode(). Kept separate from evaluate_policy.py
so metrics can be reused elsewhere (e.g. notebooks, the paper's results
generation) without re-running rollouts.

Each function takes one episode dict (or a list of them for success_rate)
with keys: obs, actions, rewards, root_heights, goal_rewards,
imitation_rewards, terminated, truncated, n_steps.
"""

import numpy as np


def episode_pose_error(episode):
    """Mean imitation pose error for one episode, in z-score units.

    reward.py defines imitation_reward = -imitation_error, so we
    recover the error directly rather than recomputing z-scores here —
    this keeps the metric consistent with whatever weighting/formula
    reward.py actually used during training.

    Lower is better (0 = perfect match to reference trajectory).
    """
    if len(episode["imitation_rewards"]) == 0:
        return np.nan
    return float(-np.mean(episode["imitation_rewards"]))


def episode_smoothness(episode):
    """Mean absolute frame-to-frame action change for one episode.

    A crude proxy for motion smoothness / jerkiness: large deltas between
    consecutive actions suggest jittery, unnatural control. This does not
    measure smoothness of the resulting *motion* (joint trajectories) —
    see pose_trajectory_smoothness() for that.

    Lower is smoother.
    """
    actions = episode["actions"]
    if len(actions) < 2:
        return np.nan
    deltas = np.diff(actions, axis=0)
    return float(np.mean(np.abs(deltas)))


def pose_trajectory_smoothness(episode, joint_indices=(0, 1, 2, 3)):
    """Mean absolute frame-to-frame change in observed joint angles.

    Complements episode_smoothness(): that measures control-signal
    jitter, this measures the resulting joint-angle jitter, which is
    closer to what a reader would perceive as a "smooth" vs "jerky"
    climbing motion. Defaults to the first four observation dims,
    matching climb_env.py's [R_hip, R_knee, L_hip, L_knee] ordering.
    """
    obs = episode["obs"][:, list(joint_indices)]
    if len(obs) < 2:
        return np.nan
    deltas = np.diff(obs, axis=0)
    return float(np.mean(np.abs(deltas)))


def fall_step(episode):
    """Step index at which the episode terminated due to falling, or None."""
    if episode["terminated"]:
        return episode["n_steps"]
    return None


def success_rate(episodes, min_steps=None):
    """Fraction of episodes that did NOT end in a fall (root height < 0.5m).

    An episode counts as a success if it was truncated (ran out of time /
    reached the step cap) rather than terminated (fell). Optionally require
    at least min_steps to also count near-misses (e.g. fell right at the
    very end) as failures even if technically truncated.
    """
    if len(episodes) == 0:
        return float("nan")

    successes = 0
    for ep in episodes:
        survived = not ep["terminated"]
        if min_steps is not None:
            survived = survived and ep["n_steps"] >= min_steps
        successes += int(survived)

    return successes / len(episodes)


def summarize(label, values):
    """Format a mean +/- std summary line, ignoring NaNs."""
    values = np.array(values, dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return f"{label}: no valid episodes"
    return f"{label}: {valid.mean():.4f} +/- {valid.std():.4f} (n={len(valid)})"
