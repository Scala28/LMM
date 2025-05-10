import torch
import numpy as np
import onnxruntime as ort
import my_modules.NNModels as NNModels
from trainings.train_common import load_network, save_network_onnx
from trainings.train_common import load_database, load_features, load_latent
import my_modules.quat_functions as quat
from my_modules import Bvh as bvh
import main_settings as ms
import struct

database = load_database('generate/data/fight/actions/database.bin')

parents = database['bone_parents']
contacts = database['contact_states']
range_starts = database['range_starts']
range_stops = database['range_stops']
X = load_features('generate/data/fight/actions/features.bin')['features'].astype(np.float32)

Ypos = database['bone_positions'].astype(np.float32)
Yrot = database['bone_rotations'].astype(np.float32)
Yvel = database['bone_velocities'].astype(np.float32)
Yang = database['bone_angular_velocities'].astype(np.float32)

# As pyTorch tensors
X = torch.as_tensor(X)

Ypos = torch.as_tensor(Ypos) # (nframes, nbones, 3/4)
Yrot = torch.as_tensor(Yrot)
Yvel = torch.as_tensor(Yvel)
Yang = torch.as_tensor(Yang)

nframes = Ypos.shape[0]
nbones = Ypos.shape[1]
nextra = contacts.shape[1]
nfeatures = X.shape[1]
nlatent = 34
dt = 1.0 / 60.0

# Compute global space
Gpos, Grot, Gvel, Gang = quat.fk_vel(Ypos, Yrot, Yvel, Yang, parents)
# Compute character space
Qpos = quat.inv_mul_vec(Grot[:, 0:1], Gpos - Gpos[:, 0:1])
Qrot = quat.inv_mul(Grot[:, 0:1], Grot)
Qvel = quat.inv_mul_vec(Grot[:, 0:1], Gvel)
Qang = quat.inv_mul_vec(Grot[:, 0:1], Gang)
# Convert to rotation matrix: (nframes, nbones, 3, 3)
Yxfm = quat.to_xform(Yrot)
Qxfm = quat.to_xform(Qrot)
# Convert to 2 axis rotation matrix: (nframes, nbones, 3, 2)
Ytxy = quat.to_xform_xy(Yrot)
Qtxy = quat.to_xform_xy(Qrot)
# Compute local root velocity
Yrvel = quat.inv_mul_vec(Yrot[:, 0], Yvel[:, 0])  # (nframes, 3)
Yrang = quat.inv_mul_vec(Yrot[:, 0], Yang[:, 0])  # (nframes, 3)
# Compute extra outputs
Yextra = torch.as_tensor(contacts.astype(np.float32))


com_mean_in, com_std_in, com_mean_out, com_std_out, com_layers = load_network('train_ris/fight/move/decompressor/compressor.bin')
compressor = NNModels.Compressor.load(com_mean_in, com_std_in, com_mean_out, com_std_out, com_layers)

decom_mean_in, decom_std_in, decom_mean_out, decom_std_out, decom_layers = load_network('train_ris/fight/move/decompressor/decompressor.bin')
decompressor = NNModels.Decompressor.load(decom_mean_in, decom_std_in, decom_mean_out, decom_std_out, decom_layers)

with torch.no_grad():
    Z = compressor((torch.cat((
        Ypos[:, 1:].reshape([1, nframes, -1]),  # (1, nframes, (bones-1)*3)
        Ytxy[:, 1:].reshape([1, nframes, -1]),  # (1, nframes, (bones-1)*3*2)
        Yvel[:, 1:].reshape([1, nframes, -1]),
        Yang[:, 1:].reshape([1, nframes, -1]),
        Qpos[:, 1:].reshape([1, nframes, -1]),
        Qtxy[:, 1:].reshape([1, nframes, -1]),
        Qvel[:, 1:].reshape([1, nframes, -1]),
        Qang[:, 1:].reshape([1, nframes, -1]),
        Yrvel.reshape([1, nframes, -1]),
        Yrang.reshape([1, nframes, -1]),
        Yextra.reshape([1, nframes, -1])
    ), dim=-1) - com_mean_in) / com_std_in)

    with open('./train_ris/{0}/{1}/decompressor/latent.bin'.format(ms.controller_type, ms.animation_type), 'wb') as f:
        f.write(struct.pack('II', nframes, nlatent) + Z.cpu().numpy().astype(np.float32).ravel().tobytes())

    start = range_starts[0]
    stop = range_stops[len(range_stops)-1]

    Ygnd_pos = Ypos[start:stop][np.newaxis]  # (1, stop-start, nbones, 3)
    Ygnd_rot = Yrot[start: stop][np.newaxis]  # (1, stop- start, nbones, 4)
    Ygnd_txy = Ytxy[start: stop][np.newaxis]  # (1, stop - start, nbones, 3, 2)
    Ygnd_vel = Yvel[start:stop][np.newaxis]
    Ygnd_ang = Yang[start:stop][np.newaxis]

    Qgnd_pos = Qpos[start:stop][np.newaxis]
    Qgnd_txy = Qtxy[start:stop][np.newaxis]
    Qgnd_vel = Qvel[start:stop][np.newaxis]
    Qgnd_ang = Qang[start:stop][np.newaxis]

    Ygnd_rvel = Yrvel[start:stop][np.newaxis]
    Ygnd_rang = Yrang[start:stop][np.newaxis]
    Ygnd_extra = Yextra[start:stop][np.newaxis]

    Xgnd = X[start:stop][np.newaxis]  # (1, stop-start, nfeatures)

    Zgnd = compressor((torch.cat([
        Ygnd_pos[:, :, 1:].reshape([1, stop - start, -1]),  # (1, stop-start, (nbones-1)*3)
        Ygnd_txy[:, :, 1:].reshape([1, stop - start, -1]),
        Ygnd_vel[:, :, 1:].reshape([1, stop - start, -1]),
        Ygnd_ang[:, :, 1:].reshape([1, stop - start, -1]),
        Qgnd_pos[:, :, 1:].reshape([1, stop - start, -1]),
        Qgnd_txy[:, :, 1:].reshape([1, stop - start, -1]),
        Qgnd_vel[:, :, 1:].reshape([1, stop - start, -1]),
        Qgnd_ang[:, :, 1:].reshape([1, stop - start, -1]),
        Ygnd_rvel.reshape([1, stop - start, -1]),
        Ygnd_rang.reshape([1, stop - start, -1]),
        Ygnd_extra.reshape([1, stop - start, -1])
    ], dim=-1) - com_mean_in) / com_std_in)

    Ytil = (decompressor(torch.cat([Xgnd[..., :-1], Zgnd], dim=-1))
            * decom_std_out + decom_mean_out)

    Ytil_pos = Ytil[:, :, 0 * (nbones - 1):3 * (nbones - 1)].reshape([1, stop - start, nbones - 1, 3])
    Ytil_txy = Ytil[:, :, 3 * (nbones - 1):9 * (nbones - 1)].reshape([1, stop - start, nbones - 1, 3, 2])
    Ytil_rvel = Ytil[:, :, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([1, stop - start, 3])
    Ytil_rang = Ytil[:, :, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([1, stop - start, 3])

    # Convert to quat and remove batch
    Ytil_rot = quat.from_xfm_xy(Ytil_txy[0])  # (stop-start, nbones-1, 4)
    Ytil_pos = Ytil_pos[0]
    Ytil_rvel = Ytil_rvel[0]  # (stop-start, 3)
    Ytil_rang = Ytil_rang[0]

    # Add root
    Ytil_rpos = [Ygnd_pos[0, 0, 0]]  # [(3,)]
    Ytil_rrot = [Ygnd_rot[0, 0, 0]]  # [(4,)]
    for i in range(1, Ygnd_pos.shape[1]):
        Ytil_rpos.append(Ytil_rpos[-1] + quat.mul_vec(Ytil_rrot[-1], Ytil_rvel[i - 1]) * dt)
        Ytil_rrot.append(quat.mul(Ytil_rrot[-1], quat.from_scaled_axis_angle(quat.mul_vec(
            Ytil_rrot[-1], Ytil_rang[i - 1]) * dt)))

    Ytil_rpos = torch.cat([p[np.newaxis] for p in Ytil_rpos])  # (stop-start, 3)
    Ytil_rrot = torch.cat([r[np.newaxis] for r in Ytil_rrot])  # (stop-start, 4)

    # Ytil_rpos = Ygnd_pos[0][:, 0:1]
    # Ytil_rrot = Ygnd_rot[0][:, 0:1]

    Ytil_pos = torch.cat([Ytil_rpos[:, np.newaxis], Ytil_pos], dim=1)  # (stop-start, nbones, 3)
    Ytil_rot = torch.cat([Ytil_rrot[:, np.newaxis], Ytil_rot], dim=1)  # (stop-start, nbones, 4)

    # Ytil_pos = torch.cat([Ytil_rpos, Ytil_pos], dim=1)
    # Ytil_rot = torch.cat([Ytil_rrot, Ytil_rot], dim=1)

    # Write BVH
    try:
        bvh.save('./train_ris/{0}/{1}/decompressor/decompressor_Ygnd.bvh'.format(ms.controller_type, ms.animation_type), {
            'rotations': np.degrees(quat.to_euler(Ygnd_rot[0].cpu().numpy())),
            'positions': 100.0 * Ygnd_pos[0].cpu().numpy(),
            'offsets': 100.0 * Ygnd_pos[0, 0].cpu().numpy(),
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'yxz'
        })
        bvh.save('./train_ris/{0}/{1}/decompressor/decompressor_Ytil.bvh'.format(ms.controller_type, ms.animation_type), {
            'rotations': np.degrees(quat.to_euler(Ytil_rot)),
            'positions': 100.0 * Ytil_pos,
            'offsets': 100.0 * Ytil_pos[0],
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'yxz'
        })
    except IOError as e:
        print(e)
