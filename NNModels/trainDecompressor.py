from torch.utils.tensorboard import SummaryWriter
import time
import torch
import my_modules.NNModels as NNModels
import my_modules.Custom as CustomFunctions

import numpy as np

# runtime start
start_time = time.time()

# checking if GPU is available
# print(torch.cuda.get_device_name(0))
device = torch.device("cpu")

# load data
X = CustomFunctions.LoadData("XData")['data']
Y = CustomFunctions.LoadData("YData")['data']
indices = CustomFunctions.LoadData("YData")['indices']

hierarchy = CustomFunctions.LoadData("HierarchyData")['data']
hierarchy = [int(hierarchy[i][0]) for i in range(len(hierarchy))]

# To tensor
X = torch.as_tensor(X, dtype=torch.float).to(device)
Y = torch.as_tensor(Y, dtype=torch.float).to(device)

# compute forward kinematics
Q = CustomFunctions.quat_forwardkinematics(Y, hierarchy)

# converting rotations:
# 9d rotation matrix (13 --> 18)
Y_9drm = torch.empty((Y.shape[0], 0), dtype=torch.float).to(device)
Q_9drm = torch.empty((Q.shape[0], 0), dtype=torch.float).to(device)

# 6d representation (13 --> 15)
Y_6dr = torch.empty((Y.shape[0], 0), dtype=torch.float).to(device)
Q_6dr = torch.empty((Q.shape[0], 0), dtype=torch.float).to(device)

for i in range(0, Y.size(1), 13):
    Y_9drm = torch.cat((Y_9drm, Y[..., i:i + 3], CustomFunctions.quat_to_9drm(Y[..., i + 3:i + 7]),
                        Y[..., i + 7: i + 13]), dim=-1)
    Q_9drm = torch.cat((Q_9drm, Q[..., i:i + 3], CustomFunctions.quat_to_9drm(Q[..., i + 3:i + 7]),
                        Q[..., i + 7: i + 13]), dim=-1)
    Y_6dr = torch.cat((Y_6dr, Y[..., i:i + 3], CustomFunctions.quat_to_6dr(Y[..., i + 3:i + 7]),
                       Y[..., i + 7:i + 13]), dim=-1)
    Q_6dr = torch.cat((Q_6dr, Q[..., i:i + 3], CustomFunctions.quat_to_6dr(Q[..., i + 3:i + 7]),
                       Q[..., i + 7:i + 13]), dim=-1)

# relevant dataset indexes
pos_6dr = []
rot_6dr = []
vel_6dr = []
ang_6dr = []

[pos_6dr.extend(list(range(i, i + 3))) for i in range(0, Y_6dr.size(1), 15)]
[rot_6dr.extend(list(range(i + 3, i + 9))) for i in range(0, Y_6dr.size(1), 15)]
[vel_6dr.extend(list(range(i + 9, i + 12))) for i in range(0, Y_6dr.size(1), 15)]
[ang_6dr.extend(list(range(i + 12, i + 15))) for i in range(0, Y_6dr.size(1), 15)]

pos_9drm = []
rot_9drm = []
vel_9drm = []
ang_9drm = []

[pos_9drm.extend(list(range(i, i + 3))) for i in range(0, Y_9drm.size(1), 18)]
[rot_9drm.extend(list(range(i + 3, i + 12))) for i in range(0, Y_9drm.size(1), 18)]
[vel_9drm.extend(list(range(i + 12, i + 15))) for i in range(0, Y_9drm.size(1), 18)]
[ang_9drm.extend(list(range(i + 15, i + 18))) for i in range(0, Y_9drm.size(1), 18)]

Ypos_scale = Y_6dr[..., pos_6dr].std()
Yrot_scale = Y_6dr[..., rot_6dr].std()
Yvel_scale = Y_6dr[..., vel_6dr].std()
Yang_scale = Y_6dr[..., ang_6dr].std()

Qpos_scale = Q_6dr[..., pos_6dr].std()
Qrot_scale = Q_6dr[..., rot_6dr].std()
Qvel_scale = Q_6dr[..., vel_6dr].std()
Qang_scale = Q_6dr[..., ang_6dr].std()

# scaling Y & Q
compressor_mean = torch.cat((Y_6dr, Q_6dr), dim=1).mean(dim=0)
compressor_std = torch.empty(0)

for i in range(0, Y_6dr.size(1), 15):
    compressor_std = torch.cat(
        (compressor_std, Ypos_scale.repeat(3), Yrot_scale.repeat(6), Yvel_scale.repeat(3), Yang_scale.repeat(3)), dim=0)

for i in range(0, Q_6dr.size(1), 15):
    compressor_std = torch.cat(
        (compressor_std, Qpos_scale.repeat(3), Qrot_scale.repeat(6), Qvel_scale.repeat(3), Qang_scale.repeat(3)), dim=0)

# Training settings
nFeatures = X.size(1)
nLatent = 32
epochs = 1
batchsize = 32
window = 2
logFreq = 500
dt = 1.0 / 60.0

decompressor_mean = Y_6dr.mean(dim=0)
decompressor_std = Y_6dr.std(dim=0) + 0.001

# Build batches respecting window size
# TODO: fix samples construction
samples = []
for i in range(len(indices)):
    for j in range(indices[i], indices[i] - window):
        samples.append(np.arange(j, j + window))
samples = torch.as_tensor(np.array(samples))

# nn models
compressor = NNModels.Compressor(Y_6dr.size(1) * 2, nLatent).to(device)
decompressor = NNModels.Decompressor(nFeatures + nLatent, Y_6dr.size(1)).to(device)

c_optimizer, c_scheduler = NNModels.TrainSettings(compressor)
d_optimizer, d_scheduler = NNModels.TrainSettings(decompressor)

# Init tensorboard
writer = SummaryWriter()

for t in range(epochs + 1):
    batch = samples[torch.randint(0, len(samples), size=[batchsize])]

    # (window, batch size, features) <- (batch size, window, features)
    Xgnd = X[batch.long()].transpose(0, 1)
    Ygnd_6dr = Y_6dr[batch.long()].transpose(0, 1)
    Qgnd_6dr = Q_6dr[batch.long()].transpose(0, 1)
    Qgnd_9drm = Q_9drm[batch.long()].transpose(0, 1)

    # Generate latent variables Z
    Zgnd = compressor((torch.cat((Ygnd_6dr, Qgnd_6dr), dim=-1) - compressor_mean) / compressor_std)

    # Reconstruct pose Y
    Ytil_6dr = decompressor(torch.cat((Xgnd, Zgnd), dim=-1)) * decompressor_std + decompressor_mean

    # 9d rotation matrix <-- 6d representation
    Ytil_9drm = torch.empty((Ytil_6dr.size(0), Ytil_6dr.size(1), 0)).to(device)
    for i in range(0, Ytil_6dr.size(2), 15):
        Ytil_9drm = torch.cat((Ytil_9drm, Ytil_6dr[..., i:i + 3],
                               CustomFunctions.rm_from_6dr(Ytil_6dr[..., i + 3:i + 9]),
                               Ytil_6dr[..., i + 9:i + 15]), dim=-1)
    # quaternion <-- 9d rotation matrix
    Ytil = torch.empty((Ytil_6dr.size(0), Ytil_6dr.size(1), 0)).to(device)
    for i in range(0, Ytil_6dr.size(2), 18):
        Ytil = torch.cat((Ytil, Ytil_9drm[..., i:i+3],
                          CustomFunctions.quat_from_9drm(Ytil_9drm[..., i+3: i+12]),
                          Ytil_9drm[..., i+12:i+18]), dim=-1)
    # Compute forward kinematics
    Qtil = CustomFunctions.quat_forwardkinematics(Ytil, hierarchy)

    # 6d representation <-- quaternion
    Qtil_6dr = torch.empty((Qtil.size(0), Qtil.size(1), 0)).to(device)
    for i in range(0, Qtil.size(2), 13):
        Qtil_6dr = torch.cat((Qtil_6dr, Qtil[..., i:i+3],
                              CustomFunctions.quat_to_6dr(Qtil[..., i+3: i+7]),
                              Qtil[..., i+7:i+13]), dim=-1)

