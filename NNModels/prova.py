import torch
import numpy as np
import onnxruntime as ort
import my_modules.NNModels as NNModels
from train_common import load_network, save_network_onnx
from train_common import load_database, load_features, load_latent
import my_modules.quat_functions as quat
import bvh


database = load_database('./data/database.bin')
X = load_features('./data/features.bin')['features'].astype(np.float32)
Z = load_latent('train_ris/decompressor/latent.bin')['latent'].astype(np.float32)

frame = database['range_starts'][1]
print(frame)
parents = database['bone_parents']

Ypos = database['bone_positions'].astype(np.float32)
Yrot = database['bone_rotations'].astype(np.float32)
Ypos = torch.as_tensor(Ypos)[frame: frame+1, ...]
Yrot = torch.as_tensor(Yrot)[frame: frame+1, ...]

X = torch.as_tensor(X.astype(np.float32))[frame:frame+1, ...]
Z = torch.as_tensor(Z.astype(np.float32))[frame:frame+1, ...]
XZ = torch.cat([X, Z], dim=1)
print(XZ.shape)


mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/decompressor/decompressor.bin')
mean_in1, std_in1, mean_out1, std_out1, layers1 = load_network('train_ris/stepper/stepper_orange.bin')

session = ort.InferenceSession('train_ris/stepper/stepper_orange.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

session2 = ort.InferenceSession('train_ris/decompressor/decompressor.onnx')
input_name2 = session2.get_inputs()[0].name
output_name2 = session2.get_outputs()[0].name

currentXZ = XZ
Rot_euler_tot = torch.zeros([0, 23, 3])
Pos_tot = torch.zeros([0, 23, 3])
with torch.no_grad():
    for i in range(200):
        delta = (torch.as_tensor(
            np.array(session.run([output_name], {input_name: currentXZ[0].numpy()}))) * std_out1 + mean_out1)
        currentXZ = currentXZ + 1.0 / 60.0 * delta
        print(currentXZ.shape)
        print(delta.shape)
        Ytil = (torch.as_tensor(np.array(
            session2.run([output_name2],
                         {input_name2: currentXZ[0].numpy()}))) * std_out + mean_out)
        print(Ytil.shape)
        nbones = 23

        dt = 1 / 60

        Ytil_pos = Ytil[:, 0 * (nbones - 1):3 * (nbones - 1)].reshape([nbones - 1, 3])
        Ytil_txy = Ytil[:, 3 * (nbones - 1):9 * (nbones - 1)].reshape([nbones - 1, 3, 2])
        Ytil_rvel = Ytil[:, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([3, ])
        Ytil_rang = Ytil[:, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([3, ])

        Ytil_quat = quat.from_xfm_xy(Ytil_txy)

        rots = Yrot[0, 0]
        rootPos = Ypos[0, 0] + quat.mul_vec(rots[np.newaxis, ...], Ytil_rvel) * dt
        rootRot = quat.mul(rots, quat.from_scaled_axis_angle(quat.mul_vec(rots, Ytil_rang) * dt))

        Pos = torch.cat([rootPos, Ytil_pos], dim=0)[np.newaxis]
        Rot = torch.cat([rootRot[np.newaxis, ...], Ytil_quat], dim=0)[np.newaxis]


        Rot_euler = np.degrees(quat.to_euler(Rot.detach().numpy()))

        Rot_euler_tot = torch.cat([Rot_euler_tot, torch.as_tensor(Rot_euler)], dim=0)
        Pos_tot = torch.cat([Pos_tot, Pos], dim=0)
        print(i)


    bvh.save('prova_frame200.bvh', {
        'rotations': Rot_euler_tot,
        'positions': Pos_tot * 100.0,
        'offsets': Pos_tot[0] * 100.0,
        'parents': parents,
        'names': ['joint_%i' % i for i in range(nbones)],
        'order': 'zyx'
    })
    try:
        pass
    except IOError as e:
        print(e)