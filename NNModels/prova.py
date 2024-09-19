import torch
import numpy as np
import onnxruntime as ort
import my_modules.NNModels as NNModels
from train_common import load_network, save_network_onnx
from train_common import load_database, load_features, load_latent
import my_modules.quat_functions as quat
import bvh

database = load_database('data/terrain_db.bin')

parents = database['bone_parents']
contacts = database['contact_states']
range_starts = database['range_starts']
range_stops = database['range_stops']
X = load_features('data/terrain_features.bin')['features'].astype(np.float32)
Z = load_latent('./train_ris/decompressor/latent.bin')['latent'].astype(np.float32)

nfeatures = X.shape[1]
nlatent = Z.shape[1]
nextra = contacts.shape[1]

clip_index = 0
start = database['range_starts'][clip_index]
stop = min(database['range_stops'][clip_index], start + 3000)

Ypos = database['bone_positions'].astype(np.float32)
Yrot = database['bone_rotations'].astype(np.float32)
Yvel = database['bone_velocities'].astype(np.float32)
Yang = database['bone_angular_velocities'].astype(np.float32)
trajectory_toe_positions = database['trajectory_toe_positions'].astype(np.float32)

nbones = Ypos.shape[1]

# As pyTorch tensors
X = torch.as_tensor(X)[start:stop]  # (nframes, nfeatures)
Z = torch.as_tensor(Z)[start:stop]

Ygnd_pos = torch.as_tensor(Ypos)[start:stop]  # (nframes, nbones, 3/4)
Ygnd_rot = torch.as_tensor(Yrot)[start:stop]
Ygnd_vel = torch.as_tensor(Yvel)[start:stop]
Ygnd_ang = torch.as_tensor(Yang)[start:stop]
Qgnd_traj_toe_pos = torch.as_tensor(trajectory_toe_positions)[start:stop]
Qgnd_terrain = torch.as_tensor(database['terrain_positions'].astype(np.float32))[start:stop].reshape(
    [stop - start, 2, 3]
)

stepper_mean_in, stepper_std_in, stepper_mean_out, stepper_std_out, stepper_layers = load_network(
    'train_ris/stepper/stepper.bin')
stepper = NNModels.Stepper.load(stepper_mean_in, stepper_std_in, stepper_mean_out, stepper_std_out, stepper_layers)

decompressor_mean_in, decompressor_std_in, decompressor_mean_out, decompressor_std_out, decompressor_layers = (
    load_network('train_ris/decompressor/decompressor.bin'))
decompressor = NNModels.Decompressor.load(decompressor_mean_in, decompressor_std_in, decompressor_mean_out,
                                          decompressor_std_out, decompressor_layers)
projector_mean_in, projector_std_in, projector_mean_out, projector_std_out, projector_layers = load_network(
    'train_ris/projector/projector.bin'
)
projector = NNModels.Projector.load(projector_mean_in, projector_std_in, projector_mean_out, projector_std_out,
                                    projector_layers)

dt = 1.0 / 60

with torch.no_grad():
    Ytil = decompressor(torch.cat([X, Z], dim=-1)) * decompressor_std_out + decompressor_mean_out

    Ytil_pos = Ytil[:, 0 * (nbones - 1):3 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])
    Ytil_txy = Ytil[:, 3 * (nbones - 1):9 * (nbones - 1)].reshape([stop - start, nbones - 1, 3, 2])
    Ytil_vel = Ytil[:, 9 * (nbones - 1):12 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])
    Ytil_ang = Ytil[:, 12 * (nbones - 1):15 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])

    Ytil_rvel = Ytil[:, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([stop - start, 3])
    Ytil_rang = Ytil[:, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([stop - start, 3])
    Ytil_extra = Ytil[:, 15 * (nbones - 1) + 6:15 * (nbones - 1) + 6 + nextra].reshape(
        [stop - start, nextra]
    )

    Qtil_traj_toe_pos = Ytil[:, 15 * (nbones - 1) + 6 + nextra:15 * (nbones - 1) + 6 + nextra + 3 * 2 * 3].reshape(
        [stop - start, 3, 2, 3]
    )
    print(start)
    print(Qgnd_traj_toe_pos[1])
    print(Qtil_traj_toe_pos[1])

    Xgnd = X[np.newaxis]
    Zgnd = Z[np.newaxis]

    Xtil = Xgnd.clone()
    Ztil = Zgnd.clone()

    for k in range(1, stop - start):
        if (k - 1) % 20 == 0:  # Simulating the Projector's goal
            Xtil_prev = Xgnd[:, k - 1]
            Ztil_prev = Zgnd[:, k - 1]
        else:
            Xtil_prev = Xtil[:, k - 1]
            Ztil_prev = Ztil[:, k - 1]

        delta = (stepper((torch.cat([Xtil_prev, Ztil_prev], dim=-1) -
                          stepper_mean_in) / stepper_std_in) *
                 stepper_std_out + stepper_mean_out)
        Xtil[:, k] = Xtil_prev + dt * delta[:, :nfeatures]
        Ztil[:, k] = Ztil_prev + dt * delta[:, nfeatures:]

    Ytil = decompressor(torch.cat([Xtil[0], Ztil[0]], dim=-1)) * decompressor_std_out + decompressor_mean_out

    Ytil_pos = Ytil[:, 0 * (nbones - 1):3 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])
    Ytil_txy = Ytil[:, 3 * (nbones - 1):9 * (nbones - 1)].reshape([stop - start, nbones - 1, 3, 2])
    Ytil_vel = Ytil[:, 9 * (nbones - 1):12 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])
    Ytil_ang = Ytil[:, 12 * (nbones - 1):15 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])

    Ytil_rvel = Ytil[:, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([stop - start, 3])
    Ytil_rang = Ytil[:, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([stop - start, 3])
    Ytil_extra = Ytil[:, 15 * (nbones - 1) + 6:15 * (nbones - 1) + 6 + nextra].reshape(
        [stop - start, nextra]
    )

    Qtil_traj_toe_pos = Ytil[:, 15 * (nbones - 1) + 6 + nextra:15 * (nbones - 1) + 6 + nextra + 3 * 2 * 3].reshape(
        [stop - start, 3, 2, 3]
    )

    print(Qtil_traj_toe_pos[1])
    print(Xtil[0][1])
