import ezc3d
import numpy as np

c = ezc3d.c3d("data/13_35.c3d")
labels = c["parameters"]["POINT"]["LABELS"]["value"]
points = c["data"]["points"]  # (4, n_markers, n_frames)
frame_rate = c["parameters"]["POINT"]["RATE"]["value"][0]
n_frames = points.shape[2]

def idx(name):
    return labels.index(f"rory7:{name}")

def joint_angle(a_name, b_name, c_name, f):
    """Angle at point b, formed by points a-b-c, at frame f (degrees)."""
    a = points[0:3, idx(a_name), f]
    b = points[0:3, idx(b_name), f]
    c_ = points[0:3, idx(c_name), f]
    v1 = a - b
    v2 = c_ - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# --- Compute angles for every frame ---
angles = {
    "R_hip":   np.zeros(n_frames),
    "R_knee":  np.zeros(n_frames),
    "R_ankle": np.zeros(n_frames),
    "L_hip":   np.zeros(n_frames),
    "L_knee":  np.zeros(n_frames),
    "L_ankle": np.zeros(n_frames),
}

for f in range(n_frames):
    angles["R_hip"][f]   = joint_angle("RBWT", "RFWT", "RKNE", f)   # torso-ish -> hip -> knee
    angles["R_knee"][f]  = joint_angle("RFWT", "RKNE", "RANK", f)
    angles["R_ankle"][f] = joint_angle("RKNE", "RANK", "RTOE", f)
    angles["L_hip"][f]   = joint_angle("LBWT", "LFWT", "LKNE", f)
    angles["L_knee"][f]  = joint_angle("LFWT", "LKNE", "LANK", f)
    angles["L_ankle"][f] = joint_angle("LKNE", "LANK", "LTOE", f)

# --- Velocities: simple frame-to-frame difference * frame rate ---
velocities = {}
for k, v in angles.items():
    vel = np.zeros(n_frames)
    vel[1:] = (v[1:] - v[:-1]) * frame_rate
    velocities[k + "_vel"] = vel

# --- Root: use pelvis marker PEL0 for height, approximate root position ---
root_idx = idx("PEL0")
root_pos = points[0:3, root_idx, :]  # shape (3, n_frames)
root_height = root_pos[2, :]  # z = height (mm)

root_vel = np.zeros((3, n_frames))
root_vel[:, 1:] = (root_pos[:, 1:] - root_pos[:, :-1]) * frame_rate

# --- Save everything into one structured file ---
output = {
    "frame_rate": frame_rate,
    "n_frames": n_frames,
    **angles,
    **velocities,
    "root_height": root_height,
    "root_vel_x": root_vel[0],
    "root_vel_y": root_vel[1],
    "root_vel_z": root_vel[2],
}

np.savez("data/processed/13_35_reference.npz", **output)
print("Saved reference trajectory to data/processed/13_35_reference.npz")
print("\nKeys saved:", list(output.keys()))
print("\nSample — right knee angle, first 5 frames:", angles["R_knee"][:5])
print("Sample — right knee velocity, first 5 frames:", velocities["R_knee_vel"][:5])