import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from envs.reward import ClimbReward

class ClimbEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 67}

    JOINT_NAMES = [
        "right_hip_y", "right_knee",
        "left_hip_y", "left_knee",
    ]

    def __init__(self, render_mode=None, difficulty=0.0, **kwargs):
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
        )
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "humanoid_climb.xml")
        )
        MujocoEnv.__init__(
            self,
            model_path=xml_path,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )

        reference_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "processed", "13_35_reference.npz")
        )
        self.reward_fn = ClimbReward(reference_path)
        self.max_episode_steps = 300
        self.current_step = 0

        # --- Curriculum settings ---
        # difficulty: 0.0 = spawn very close to goal (easy), 1.0 = spawn at true start (hard)
        self.difficulty = difficulty
        self.true_start_x = 0.0   # the real starting x position, far from the goal
        self.near_goal_x = 0.85   # a position close to the goal, for easy mode

    def set_difficulty(self, value):
        """Call this externally to ramp difficulty up over training, e.g. env.set_difficulty(0.3)"""
        self.difficulty = float(np.clip(value, 0.0, 1.0))

    def _quat_to_pitch_roll(self, quat):
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return pitch, roll

    def _get_obs(self):
        joint_angles = []
        joint_vels = []
        for name in self.JOINT_NAMES:
            qpos_addr = self.model.joint(name).qposadr[0]
            qvel_addr = self.model.joint(name).dofadr[0]
            joint_angles.append(self.data.qpos[qpos_addr])
            joint_vels.append(self.data.qvel[qvel_addr])

        root_height = self.data.qpos[2]
        root_vel = self.data.qvel[0:3]

        quat = self.data.qpos[3:7]
        pitch, roll = self._quat_to_pitch_roll(quat)

        right_foot_id = self.model.geom("right_foot").id
        left_foot_id = self.model.geom("left_foot").id
        right_contact = 0.0
        left_contact = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = (contact.geom1, contact.geom2)
            if right_foot_id in geoms:
                right_contact = 1.0
            if left_foot_id in geoms:
                left_contact = 1.0

        ankle_placeholder = [0.0, 0.0, 0.0, 0.0]

        obs = np.array(
            joint_angles + joint_vels +
            ankle_placeholder +
            [root_height] +
            list(root_vel) +
            [pitch, roll] +
            [right_contact, left_contact],
            dtype=np.float64
        )
        return obs

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # --- Curriculum: interpolate starting x position based on difficulty ---
        start_x = self.near_goal_x + self.difficulty * (self.true_start_x - self.near_goal_x)
        # add a little randomness so it's not the exact same spawn every episode
        start_x += np.random.uniform(-0.03, 0.03)
        qpos[0] = start_x

        self.set_state(qpos, qvel)
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        self.current_step += 1

        root_pos = self.data.qpos[0:3]
        right_knee = self.data.qpos[self.model.joint("right_knee").qposadr[0]]
        left_knee = self.data.qpos[self.model.joint("left_knee").qposadr[0]]
        right_hip = self.data.qpos[self.model.joint("right_hip_y").qposadr[0]]
        left_hip = self.data.qpos[self.model.joint("left_hip_y").qposadr[0]]

        episode_progress = min(self.current_step / self.max_episode_steps, 1.0)
        reward, goal_r, imit_r = self.reward_fn.compute(
            root_pos, right_knee, left_knee, right_hip, left_hip, episode_progress
        )

        root_height = self.data.qpos[2]
        terminated = bool(root_height < 0.5)
        truncated = bool(self.current_step >= self.max_episode_steps)

        # --- Fall penalty: one-time cost for collapsing, discourages diving toward the goal ---
        if terminated:
            reward -= 10.0

        info = {"goal_reward": goal_r, "imitation_reward": imit_r}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = ClimbEnv(difficulty=0.0)  # start easy
    obs, info = env.reset()
    print("Easy mode start x:", env.data.qpos[0])

    env.set_difficulty(1.0)  # now try hard mode
    obs, info = env.reset()
    print("Hard mode start x:", env.data.qpos[0])

    env.close()