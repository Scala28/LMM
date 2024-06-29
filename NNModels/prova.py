import torch
import numpy as np
import my_modules.NNModels as NNModels
from train_common import load_network, save_network_onnx
from train_common import load_database, load_features, load_latent
import my_modules.quat_functions as quat

# mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/decompressor/decompressor.bin')
# print(mean_in.shape)
# print(mean_out.shape)
# decompressor = NNModels.Decompressor.load(mean_in, std_in, mean_out, std_out, layers)
# save_network_onnx(decompressor, torch.as_tensor(mean_in.copy()), 'train_ris/decompressor/decompressor.onnx')

database = load_database('./data/database.bin')
X = load_features('./data/features.bin')['features'].astype(np.float32)
Z = load_latent('train_ris/decompressor/latent.bin')['latent'].astype(np.float32)

X = torch.as_tensor(X)[2:3, ...]
Z = torch.as_tensor(Z)[2:3, ...]
XZ = torch.cat([X, Z], dim=1)
print(XZ.shape)


mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/decompressor/decompressor.bin')

dec = NNModels.Decompressor(mean_in, mean_out, layers)

Ytil = dec(XZ)
nbones = 23

dt = 1/60

Ytil_pos = Ytil[:, 0 * (nbones - 1):3 * (nbones - 1)].reshape([nbones - 1, 3])
Ytil_txy = Ytil[:, 3 * (nbones - 1):9 * (nbones - 1)].reshape([nbones - 1, 3, 2])
Ytil_rvel = Ytil[:, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([3, ])
Ytil_rang = Ytil[:, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([3, ])

Ytil_quat = quat.from_xfm_xy(Ytil_txy)


rots = torch.as_tensor(np.array([0, 0, 0, 1], dtype=np.float32).reshape(4,))
rootPos = torch.zeros([3,]) + quat.mul_vec(rots[np.newaxis, ...], Ytil_rvel) * dt
rootRot = quat.mul(rots, quat.from_scaled_axis_angle(quat.mul_vec(rots, Ytil_rang) * dt))

Pos = torch.cat([rootPos, Ytil_pos], dim=0)
Rot_quat = torch.cat([rootRot[np.newaxis, ...], Ytil_quat], dim=0)

print(Pos)
print(Rot_quat)
Rot_euler = quat.to_euler(Rot_quat.detach().numpy())
print(Rot_euler)



