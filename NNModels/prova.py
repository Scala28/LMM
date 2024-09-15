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

clip_index = 6
start = database['range_starts'][clip_index]
stop = min(database['range_stops'][clip_index], start + 3000)

Ypos = database['bone_positions'].astype(np.float32)
Yrot = database['bone_rotations'].astype(np.float32)
Yvel = database['bone_velocities'].astype(np.float32)
Yang = database['bone_angular_velocities'].astype(np.float32)

nbones = Ypos.shape[1]

# As pyTorch tensors
X = torch.as_tensor(X)[start:stop]  # (nframes, nfeatures)
Z = torch.as_tensor(Z)[start:stop]
print(X.shape)
print(Z.shape)

Ygnd_pos = torch.as_tensor(Ypos)[start:stop]  # (nframes, nbones, 3/4)
Ygnd_rot = torch.as_tensor(Yrot)[start:stop]
Ygnd_vel = torch.as_tensor(Yvel)[start:stop]
Ygnd_ang = torch.as_tensor(Yang)[start:stop]

Qgnd_terrain = torch.as_tensor(database['terrain_positions'].astype(np.float32))[start:stop].reshape(
    [stop - start, 2, 3]
)

# Compute global space
Gpos, Grot, Gvel, Gang = quat.fk_vel(Ygnd_pos, Ygnd_rot, Ygnd_vel, Ygnd_ang, parents)
# Compute character space
Qpos = quat.inv_mul_vec(Grot[:, 0:1], Gpos - Gpos[:, 0:1])
Qtoe_pos = torch.cat([Qpos[:, 5:6], Qpos[:, 9:10]], dim=1)

Qtraj_toe_pos = torch.as_tensor(database['trajectory_toe_positions'].astype(np.float32))[start:stop]

stepper_mean_in, stepper_std_in, stepper_mean_out, stepper_std_out, stepper_layers = load_network(
    'train_ris/stepper/stepper.bin')
stepper = NNModels.Stepper.load(stepper_mean_in, stepper_std_in, stepper_mean_out, stepper_std_out, stepper_layers)

decompressor_mean_in, decompressor_std_in, decompressor_mean_out, decompressor_std_out, decompressor_layers = (
    load_network('train_ris/decompressor/decompressor.bin'))
decompressor = NNModels.Decompressor.load(decompressor_mean_in, decompressor_std_in, decompressor_mean_out,
                                          decompressor_std_out, decompressor_layers)

dt = 1.0 / 60

with torch.no_grad():
    Xgnd = X[np.newaxis]
    Zgnd = Z[np.newaxis]

    Xtil = Xgnd.clone()
    Ztil = Zgnd.clone()

    Ytil = torch.zeros([1, Xgnd.shape[1], decompressor_mean_out.shape[0]])

    for k in range(1, stop - start):
        if (k - 1) % 20 == 0:
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

        Ytil[:, k] = decompressor(
            torch.cat([Xtil_prev, Ztil_prev], dim=-1)
        ) * decompressor_std_out + decompressor_mean_out

    Ytil_pos = Ytil[:, :, 0 * (nbones - 1):3 * (nbones - 1)].reshape([1, stop - start, nbones - 1, 3])
    Ytil_txy = Ytil[:, :, 3 * (nbones - 1):9 * (nbones - 1)].reshape([1, stop - start, nbones - 1, 3, 2])
    Ytil_rvel = Ytil[:, :, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([1, stop - start, 3])
    Ytil_rang = Ytil[:, :, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([1, stop - start, 3])
    Ytil_traj_toe_pos = Ytil[:, :, 15 * (nbones - 1) + 6 + nextra: 15 * (nbones - 1) + 6 + nextra + 18]

    # Convert to quat and remove batch
    Ytil_rot = quat.from_xfm_xy(Ytil_txy[0])  # (stop-start, nbones-1, 4)
    Ytil_pos = Ytil_pos[0]
    Ytil_rvel = Ytil_rvel[0]  # (stop-start, 3)
    Ytil_rang = Ytil_rang[0]

    Ytil_rpos = [Ygnd_pos[0, 0]]  # [(3,)]
    Ytil_rrot = [Ygnd_rot[0, 0]]  # [(4,)]
    for i in range(1, Ygnd_pos.shape[0]):
        Ytil_rpos.append(Ytil_rpos[-1] + quat.mul_vec(Ytil_rrot[-1], Ytil_rvel[i - 1]) * dt)
        Ytil_rrot.append(quat.mul(Ytil_rrot[-1], quat.from_scaled_axis_angle(quat.mul_vec(
            Ytil_rrot[-1], Ytil_rang[i - 1]) * dt)))

    Ytil_rpos = torch.cat([p[np.newaxis] for p in Ytil_rpos])  # (stop-start, 3)
    Ytil_rrot = torch.cat([r[np.newaxis] for r in Ytil_rrot])  # (stop-start, 4)

    Ytil_pos = torch.cat([Ytil_rpos[:, np.newaxis], Ytil_pos], dim=1)  # (stop-start, nbones, 3)
    Ytil_rot = torch.cat([Ytil_rrot[:, np.newaxis], Ytil_rot], dim=1)  # (stop-start, nbones, 4)

    try:
        bvh.save('train_ris/prova/stepper_terrain_%2i_gnd.bvh' % clip_index, {
            'rotations': np.degrees(quat.to_euler(Ygnd_rot.cpu().numpy())),
            'positions': 100.0 * Ygnd_pos.cpu().numpy(),
            'offsets': 100.0 * Ygnd_pos[0].cpu().numpy(),
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'zyx'
        })
        bvh.save('train_ris/prova/stepper_terrain_%2i_til.bvh' % clip_index, {
            'rotations': np.degrees(quat.to_euler(Ytil_rot)),
            'positions': 100.0 * Ytil_pos,
            'offsets': 100.0 * Ytil_pos[1],
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'zyx'
        })
    except IOError as e:
        print(e)



