import numpy as np

data = np.load('/data/fhr/kubric/new_output/movi_physics_smoke/00003/physics.npz', allow_pickle=True)
print("Keys:", list(data.keys()))
print("c_force_raw shape:", data['c_force_raw'].shape)  # (T, N, 6)
print("c_floor:", data['c_floor'])
print("c_mass:", data['c_mass'])
print("c_static:", data['c_static'])

# 检查力是否非零
force_norm = np.linalg.norm(data['c_force_raw'][:, :, :3], axis=2).max(axis=1)
print("Max force per frame:", force_norm)
print("Contact frames:", np.where(force_norm > 0.1)[0])