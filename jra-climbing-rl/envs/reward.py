import numpy as np

class ClimbReward:
    def __init__(self, reference_path):
        data = np.load(reference_path)
        self.frame_rate = float(data["frame_rate"])
        self.n_frames = int(data["n_frames"])

        self.ref_R_knee = data["R_knee"]
        self.ref_L_knee = data["L_knee"]
        self.ref_R_hip = data["R_hip"]
        self.ref_L_hip = data["L_hip"]

        self.R_knee_mean, self.R_knee_std = self.ref_R_knee.mean(), self.ref_R_knee.std()
        self.L_knee_mean, self.L_knee_std = self.ref_L_knee.mean(), self.ref_L_knee.std()
        self.R_hip_mean, self.R_hip_std = self.ref_R_hip.mean(), self.ref_R_hip.std()
        self.L_hip_mean, self.L_hip_std = self.ref_L_hip.mean(), self.ref_L_hip.std()

        self.goal_pos = np.array([1.0, 0.0, 1.4])

    def _zscore(self, value, mean, std):
        if std < 1e-6:
            return 0.0
        return (value - mean) / std

    def compute(self, root_pos, right_knee_rad, left_knee_rad, right_hip_rad, left_hip_rad, episode_progress):
        dist_to_goal = np.linalg.norm(root_pos - self.goal_pos)
        goal_reward = -dist_to_goal

        ref_frame = int(episode_progress * (self.n_frames - 1))
        ref_frame = np.clip(ref_frame, 0, self.n_frames - 1)

        sim_R_knee_deg = np.degrees(right_knee_rad)
        sim_L_knee_deg = np.degrees(left_knee_rad)
        sim_R_hip_deg = np.degrees(right_hip_rad)
        sim_L_hip_deg = np.degrees(left_hip_rad)

        z_sim_R_knee = self._zscore(sim_R_knee_deg, self.R_knee_mean, self.R_knee_std)
        z_ref_R_knee = self._zscore(self.ref_R_knee[ref_frame], self.R_knee_mean, self.R_knee_std)
        z_sim_L_knee = self._zscore(sim_L_knee_deg, self.L_knee_mean, self.L_knee_std)
        z_ref_L_knee = self._zscore(self.ref_L_knee[ref_frame], self.L_knee_mean, self.L_knee_std)
        z_sim_R_hip = self._zscore(sim_R_hip_deg, self.R_hip_mean, self.R_hip_std)
        z_ref_R_hip = self._zscore(self.ref_R_hip[ref_frame], self.R_hip_mean, self.R_hip_std)
        z_sim_L_hip = self._zscore(sim_L_hip_deg, self.L_hip_mean, self.L_hip_std)
        z_ref_L_hip = self._zscore(self.ref_L_hip[ref_frame], self.L_hip_mean, self.L_hip_std)

        imitation_error = (
            abs(z_sim_R_knee - z_ref_R_knee) +
            abs(z_sim_L_knee - z_ref_L_knee) +
            abs(z_sim_R_hip - z_ref_R_hip) +
            abs(z_sim_L_hip - z_ref_L_hip)
        )
        imitation_reward = -imitation_error

        # --- Survival bonus: now scaled by proximity to goal, so standing still
        # stops being a free reward — the agent must combine staying upright
        # WITH making progress toward the goal to earn it.
        root_height = root_pos[2]
        if root_height > 1.0:
            # dist_to_goal is already computed above; scale bonus inversely with it
            proximity_scale = max(0.0, 1.0 - (dist_to_goal / 1.5))  # 1.5 = rough max relevant distance
            survival_bonus = 0.3 * proximity_scale
        else:
            survival_bonus = 0.0

        total_reward = 1.0 * goal_reward + 0.02 * imitation_reward + survival_bonus
        return total_reward, goal_reward, imitation_reward