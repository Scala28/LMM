import torch
import numpy as np
import onnxruntime as ort
import my_modules.NNModels as NNModels
from train_common import load_network, save_network_onnx
from train_common import load_database, load_features, load_latent
import my_modules.quat_functions as quat
import bvh


database = load_database('./data/database.bin')

parents = database['bone_parents']
contacts = database['contact_states']
range_starts = database['range_starts']
range_stops = database['range_stops']
X = load_features('./data/features.bin')['features'].astype(np.float32)
Z = load_latent('./train_ris/decompressor/latent.bin')['latent'].astype(np.float32)

start = database['range_starts'][2]
stop = min(database['range_stops'][2], start + 1000)

Ypos = database['bone_positions'].astype(np.float32)
Yrot = database['bone_rotations'].astype(np.float32)
Yvel = database['bone_velocities'].astype(np.float32)
Yang = database['bone_angular_velocities'].astype(np.float32)

# As pyTorch tensors
X = torch.as_tensor(X)[start:stop]  # (nframes, nfeatures)
Z = torch.as_tensor(Z)[start:stop]
Ygnd_pos = torch.as_tensor(Ypos)[start:stop]  # (nframes, nbones, 3/4)
Ygnd_rot = torch.as_tensor(Yrot)[start:stop]
Ygnd_vel = torch.as_tensor(Yvel)[start:stop]
Ygnd_ang = torch.as_tensor(Yang)[start:stop]

nframes = Ypos.shape[0]
nbones = Ypos.shape[1]
nextra = contacts.shape[1]
nfeatures = X.shape[1]
nlatent = 32


dt = 1.0 / 60.0
window = 20

mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/decompressor/decompressor.bin')
decompressor = NNModels.Decompressor.load(mean_in, std_in, mean_out, std_out, layers)

mean_in2, std_in2, mean_out2, std_out2, layers2 = load_network('train_ris/stepper/stepper_orange.bin')
stepper = NNModels.Stepper.load(mean_in2, std_in2, mean_out2, std_out2, layers2)

with torch.no_grad():
    '''
    input = torch.cat([X, Z], dim=-1)
    print(input.shape)
    
    Ytil = decompressor(input) * std_out + mean_out
    print(Ytil.shape)
    Ytil_pos = Ytil[:, 0 * (nbones - 1):3 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])
    Ytil_txy = Ytil[:, 3 * (nbones - 1):9 * (nbones - 1)].reshape([stop - start, nbones - 1, 3, 2])
    Ytil_rvel = Ytil[:, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([stop - start, 3])
    Ytil_rang = Ytil[:, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([stop - start, 3])

    # Convert to quat and remove batch
    Ytil_rot = quat.from_xfm_xy(Ytil_txy)  # (stop-start, nbones-1, 4)

    # Add root
    Ytil_rpos = [Ygnd_pos[0, 0]]  # [(3,)]
    Ytil_rrot = [Ygnd_rot[0, 0]]  # [(4,)]
    for i in range(1, Ygnd_pos.shape[0]):
        Ytil_rpos.append(Ytil_rpos[-1] + quat.mul_vec(Ytil_rrot[-1], Ytil_rvel[i - 1]) * dt)
        Ytil_rrot.append(quat.mul(Ytil_rrot[-1], quat.from_scaled_axis_angle(quat.mul_vec(
            Ytil_rrot[-1], Ytil_rang[i - 1]) * dt)))

    Ytil_rpos = torch.cat([p[np.newaxis] for p in Ytil_rpos])  # (stop-start, 3)
    Ytil_rrot = torch.cat([r[np.newaxis] for r in Ytil_rrot])  # (stop-start, 4)
    print(Ytil_pos.shape)
    Ytil_pos = torch.cat([Ytil_rpos[:, np.newaxis], Ytil_pos], dim=1)  # (stop-start, nbones, 3)
    Ytil_rot = torch.cat([Ytil_rrot[:, np.newaxis], Ytil_rot], dim=1)  # (stop-start, nbones, 4)

    try:
        bvh.save('anim.bvh', {
            'rotations': np.degrees(quat.to_euler(Ygnd_rot.cpu().numpy())),
            'positions': 100.0 * Ygnd_pos.cpu().numpy(),
            'offsets': 100.0 * Ygnd_pos[0].cpu().numpy(),
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'zyx'
        })
        bvh.save('decompr.bvh', {
            'rotations': np.degrees(quat.to_euler(Ytil_rot)),
            'positions': 100.0 * Ytil_pos,
            'offsets': 100.0 * Ytil_pos[0],
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'zyx'
        })
    except IOError as e:
        print(e)
    '''
    X = X[np.newaxis]
    Z = Z[np.newaxis]
    Xtil = X.clone()
    Ztil = Z.clone()

    for k in range(1, stop-start):
        if (k - 1) % window == 0:  # Simulating the Projector's goal
            Xtil_prev = X[:, k - 1]
            Ztil_prev = Z[:, k - 1]
        else:
            Xtil_prev = Xtil[:, k - 1]
            Ztil_prev = Ztil[:, k - 1]

        delta = (stepper((torch.cat([Xtil_prev, Ztil_prev], dim=-1) -
                          mean_in2) / std_in2) *
                 std_out2 + mean_out2)
        Xtil[:, k] = Xtil_prev + dt * delta[:, :nfeatures]
        Ztil[:, k] = Ztil_prev + dt * delta[:, nfeatures:]

    input = torch.cat([Xtil, Ztil], dim=-1)[0]
    print(input.shape)
    Ytil = decompressor(input) * std_out + mean_out
    print(Ytil.shape)
    Ytil_pos = Ytil[:, 0 * (nbones - 1):3 * (nbones - 1)].reshape([stop - start, nbones - 1, 3])
    Ytil_txy = Ytil[:, 3 * (nbones - 1):9 * (nbones - 1)].reshape([stop - start, nbones - 1, 3, 2])
    Ytil_rvel = Ytil[:, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([stop - start, 3])
    Ytil_rang = Ytil[:, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([stop - start, 3])

    # Convert to quat and remove batch
    Ytil_rot = quat.from_xfm_xy(Ytil_txy)  # (stop-start, nbones-1, 4)

    # Add root
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
        bvh.save('stepper.bvh', {
            'rotations': np.degrees(quat.to_euler(Ytil_rot)),
            'positions': 100.0 * Ytil_pos,
            'offsets': 100.0 * Ytil_pos[0],
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'zyx'
        })
    except IOError as e:
        print(e)