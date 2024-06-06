import sys
import struct
import matplotlib.pyplot as plt
import bvh

import numpy as np
import torch

import my_modules.NNModels as NNModels
import my_modules.quat_functions as quat
import my_modules.xform_functions as xform
from train_common import load_database, load_features, save_network

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # Load data
    database = load_database('./data/database.bin')

    parents = database['bone_parents']
    contacts = database['contact_states']
    range_starts = database['range_starts']
    range_stops = database['range_stops']

    X = load_features('./data/features.bin')['features'].astype(np.float32)
    Ypos = database['bone_positions'].astype(np.float32)
    Yrot = database['bone_rotations'].astype(np.float32)
    Yvel = database['bone_velocities'].astype(np.float32)
    Yang = database['bone_angular_velocities'].astype(np.float32)

    # As pyTorch tensors: (nframes, nbones, 3/4)
    X = torch.as_tensor(X)
    Ypos = torch.as_tensor(Ypos)
    Yrot = torch.as_tensor(Yrot)
    Yvel = torch.as_tensor(Yvel)
    Yang = torch.as_tensor(Yang)

    nframes = Ypos.shape[0]
    nbones = Ypos.shape[1]
    nextra = contacts.shape[1]
    nfeatures = X.shape[1]
    nlatent = 32

    # Parameters

    seed = 1234
    batchsize = 32
    lr = 0.001
    niter = 500000
    window = 2
    dt = 1.0 / 60.0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    # Compute global space
    Gpos, Grot, Gvel, Gang = quat.fk(Ypos, Yrot, Yvel, Yang, parents)

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

    # Compute mean/stds
    Ypos_scale = Ypos[:, 1:].std()
    Ytxy_scale = Ytxy[:, 1:].std()
    Yvel_scale = Yvel[:, 1:].std()
    Yang_scale = Yang[:, 1:].std()

    Qpos_scale = Qpos[:, 1:].std()
    Qtxy_scale = Qtxy[:, 1:].std()
    Qvel_scale = Qvel[:, 1:].std()
    Qang_scale = Qang[:, 1:].std()

    Yrvel_scale = Yrvel.std()
    Yrang_scale = Yrang.std()

    Yextra_scale = Yextra.std()

    decompressor_mean_out = torch.cat((
        torch.ravel(Ypos[:, 1:].mean(dim=0)),
        torch.ravel(Ytxy[:, 1:].mean(dim=0)),
        torch.ravel(Yvel[:, 1:].mean(dim=0)),
        torch.ravel(Yang[:, 1:].mean(dim=0)),
        torch.ravel(Yrvel.mean(dim=0)),
        torch.ravel(Yrang.mean(dim=0)),
        torch.ravel(Yextra.mean(dim=0))
    ))
    decompressor_std_out = torch.cat((
        torch.ravel(Ypos[:, 1:].std(dim=0)),
        torch.ravel(Ytxy[:, 1:].std(dim=0)),
        torch.ravel(Yvel[:, 1:].std(dim=0)),
        torch.ravel(Yang[:, 1:].std(dim=0)),
        torch.ravel(Yrvel.std(dim=0)),
        torch.ravel(Yrang.std(dim=0)),
        torch.ravel(Yextra.std(dim=0))
    ))

    decompressor_mean_in = torch.zeros([nfeatures + nlatent], dtype=torch.float32)
    decompressor_std_in = torch.ones([nfeatures + nlatent], dtype=torch.float32)

    compressor_mean_in = torch.cat((
        torch.ravel(Ypos[:, 1:].mean(dim=0)),
        torch.ravel(Ytxy[:, 1:].mean(dim=0)),
        torch.ravel(Yvel[:, 1:].mean(dim=0)),
        torch.ravel(Yang[:, 1:].mean(dim=0)),
        torch.ravel(Qpos[:, 1:].mean(dim=0)),
        torch.ravel(Qtxy[:, 1:].mean(dim=0)),
        torch.ravel(Qvel[:, 1:].mean(dim=0)),
        torch.ravel(Qang[:, 1:].mean(dim=0)),
        torch.ravel(Yrvel.mean(dim=0)),
        torch.ravel(Yrang.mean(dim=0)),
        torch.ravel(Yextra.mean(dim=0))
    ))
    compressor_std_in = torch.cat((
        Ypos_scale.repeat((nbones - 1) * 3),
        Ytxy_scale.repeat((nbones - 1) * 6),
        Yvel_scale.repeat((nbones - 1) * 3),
        Yang_scale.repeat((nbones - 1) * 3),
        Qpos_scale.repeat((nbones - 1) * 3),
        Qtxy_scale.repeat((nbones - 1) * 6),
        Qvel_scale.repeat((nbones - 1) * 3),
        Qang_scale.repeat((nbones - 1) * 3),
        Yrvel_scale.repeat(3),
        Yrang_scale.repeat(3),
        Yextra_scale.repeat(nextra)
    ))

    # Networks model
    compressor = NNModels.Compressor(len(compressor_mean_in), nlatent)
    decompressor = NNModels.Decompressor(nfeatures + nlatent, len(decompressor_mean_out))


    def _save_compressed_database():
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
            ), dim=-1) - compressor_mean_in) / compressor_std_in)

            with open('train_ris/latent.bin', 'wb') as f:
                f.write(struct.pack('II', nframes, nlatent) + Z.cpu().numpy().astype(np.float32).ravel().tobytes())


    def _generate_anim():
        with torch.no_grad():
            start = range_starts[2]
            stop = min(start + 1000, range_stops[2])

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
            ], dim=-1) - compressor_mean_in) / compressor_std_in)

            Ytil = (decompressor(torch.cat([Xgnd, Zgnd], dim=-1))
                    * decompressor_std_out + decompressor_mean_out)

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
                bvh.save('train_ris/decompressor_Ygnd.bvh', {
                    'rotations': np.degrees(quat.to_euler(Ygnd_rot[0].cpu().numpy())),
                    'positions': 100.0 * Ygnd_pos[0].cpu().numpy(),
                    'offsets': 100.0 * Ygnd_pos[0, 0].cpu().numpy(),
                    'parents': parents,
                    'names': ['joint_%i' % i for i in range(nbones)],
                    'order': 'zyx'
                })
                bvh.save('train_ris/decompressor_Ytil.bvh', {
                    'rotations': np.degrees(quat.to_euler(Ytil_rot)),
                    'positions': 100.0 * Ytil_pos,
                    'offsets': 100.0 * Ytil_pos[0],
                    'parents': parents,
                    'names': ['joint_%i' % i for i in range(nbones)],
                    'order': 'zyx'
                })
            except IOError as e:
                print(e)

            # Write features
            fmin, fmax = Xgnd.cpu().numpy().min(), Xgnd.cpu().numpy().max()

            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2 * nfeatures))
            for i in range(nfeatures):
                axs[i].plot(Xgnd[0, :500, i].cpu().numpy())
                axs[i].set_ylim(fmin, fmax)
            plt.tight_layout()

            try:
                plt.savefig('train_ris/decompressor_X.png')
            except IOError as e:
                print(e)
            plt.close()

            # Write latent
            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()

            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2 * nlatent))
            for i in range(nlatent):
                axs[i].plot(Zgnd[0, :500, i].cpu().numpy())
                axs[i].set_ylim(lmin, lmax)
            plt.tight_layout()

            try:
                plt.savefig('train_ris/decompressor_Z.png')
            except IOError as e:
                print(e)

            plt.close()


    # Build batches respecting window size
    indices = []
    for i in range(nframes - window + 1):
        indices.append(np.arange(i, i + window))
    indices = torch.as_tensor(np.array(indices), dtype=torch.long)

    # Train
    writer = SummaryWriter()

    # c_optimizer, c_scheduler = NNModels.TrainSettings(compressor)
    # d_optimizer, d_scheduler = NNModels.TrainSettings(decompressor)
    optimizer = torch.optim.AdamW(list(compressor.parameters()) +
                                  list(decompressor.parameters()),
                                  lr=lr,
                                  amsgrad=True,
                                  weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    rolling_loss = None
    sys.stdout.write('\n')

    for i in range(niter):
        # c_optimizer.zero_grad()
        # d_optimizer.zero_grad()
        optimizer.zero_grad()

        # Extract batch
        batch = indices[torch.randint(0, len(indices), size=[batchsize])]  # (batchsize, window)

        Xgnd = X[batch]  # (batchsize, window, nfeatures)

        Ygnd_pos = Ypos[batch]  # (batchsize, window, nbones, 3)
        Ygnd_txy = Ytxy[batch]  # (batchsize, window, nbones, 3, 2)
        Ygnd_vel = Yvel[batch]
        Ygnd_ang = Yang[batch]

        Qgnd_pos = Qpos[batch]
        Qgnd_xfm = Qxfm[batch]
        Qgnd_txy = Qtxy[batch]
        Qgnd_vel = Qvel[batch]
        Qgnd_ang = Qang[batch]

        Ygnd_rvel = Yrvel[batch]  # (batchsize, window, 3)
        Ygnd_rang = Yrang[batch]

        Ygnd_extra = Yextra[batch]

        # Encode
        Zgnd = compressor((torch.cat([
            Ygnd_pos[:, :, 1:].reshape([batchsize, window, -1]),  # (batchsize, window, (bones-1)*3)
            Ygnd_txy[:, :, 1:].reshape([batchsize, window, -1]),  # (batchsize, window, (bones-1)*3*2)
            Ygnd_vel[:, :, 1:].reshape([batchsize, window, -1]),
            Ygnd_ang[:, :, 1:].reshape([batchsize, window, -1]),
            Qgnd_pos[:, :, 1:].reshape([batchsize, window, -1]),
            Qgnd_txy[:, :, 1:].reshape([batchsize, window, -1]),
            Qgnd_vel[:, :, 1:].reshape([batchsize, window, -1]),
            Qgnd_ang[:, :, 1:].reshape([batchsize, window, -1]),
            Ygnd_rvel.reshape([batchsize, window, -1]),
            Ygnd_rang.reshape([batchsize, window, -1]),
            Ygnd_extra.reshape([batchsize, window, -1])
        ], dim=-1) - compressor_mean_in) / compressor_std_in)

        # Decode
        Ytil = (decompressor(torch.cat([Xgnd, Zgnd], dim=-1)) *
                decompressor_std_out + decompressor_mean_out)

        Ytil_pos = Ytil[:, :, 0 * (nbones - 1):3 * (nbones - 1)].reshape([batchsize, window, nbones - 1, 3])
        Ytil_txy = Ytil[:, :, 3 * (nbones - 1):9 * (nbones - 1)].reshape([batchsize, window, nbones - 1, 3, 2])
        Ytil_vel = Ytil[:, :, 9 * (nbones - 1):12 * (nbones - 1)].reshape([batchsize, window, nbones - 1, 3])
        Ytil_ang = Ytil[:, :, 12 * (nbones - 1):15 * (nbones - 1)].reshape([batchsize, window, nbones - 1, 3])

        Ytil_rvel = Ytil[:, :, 15 * (nbones - 1) + 0:15 * (nbones - 1) + 3].reshape([batchsize, window, 3])
        Ytil_rang = Ytil[:, :, 15 * (nbones - 1) + 3:15 * (nbones - 1) + 6].reshape([batchsize, window, 3])
        Ytil_extra = Ytil[:, :, 15 * (nbones - 1) + 6:15 * (nbones - 1) + 6 + nextra].reshape(
            [batchsize, window, nextra])

        # Add root bone
        Ytil_pos = torch.cat([Ygnd_pos[:, :, 0:1], Ytil_pos], dim=2)
        Ytil_txy = torch.cat([Ygnd_txy[:, :, 0:1], Ytil_txy], dim=2)
        Ytil_vel = torch.cat([Ygnd_vel[:, :, 0:1], Ytil_vel], dim=2)
        Ytil_ang = torch.cat([Ygnd_ang[:, :, 0:1], Ytil_ang], dim=2)

        # Compute forward kinematics
        Ytil_xfm = xform.from_xy(Ytil_txy)  # (bathsize, window, nbones, 3, 3)

        Gtil_pos, Gtil_xfm, Gtil_vel, Gtil_ang = xform.fk(
            Ytil_pos, Ytil_xfm, Ytil_vel, Ytil_ang, parents
        )

        # Compute character space
        Qtil_pos = xform.inv_mul_vec(Gtil_xfm[:, :, 0:1], Gtil_pos - Gtil_pos[:, :, 0:1])
        Qtil_xfm = xform.inv_mul(Gtil_xfm[:, :, 0:1], Gtil_xfm)
        Qtil_vel = xform.inv_mul_vec(Gtil_xfm[:, :, 0:1], Gtil_vel)
        Qtil_ang = xform.inv_mul_vec(Gtil_xfm[:, :, 0:1], Gtil_ang)

        # Compute deltas for positions and rotations
        Ygnd_dpos = (Ygnd_pos[:, 1:] - Ygnd_pos[:, :-1]) / dt  # frame1 - frame0 of window
        Ygnd_dtxy = (Ygnd_txy[:, 1:] - Ygnd_txy[:, :-1]) / dt
        Qgnd_dpos = (Qgnd_pos[:, 1:] - Qgnd_pos[:, :-1]) / dt
        Qgnd_dxfm = (Qgnd_xfm[:, 1:] - Qgnd_xfm[:, :-1]) / dt

        Ytil_dpos = (Ytil_pos[:, 1:] - Ytil_pos[:, :-1]) / dt
        Ytil_dtxy = (Ytil_txy[:, 1:] - Ytil_txy[:, :-1]) / dt
        Qtil_dpos = (Qtil_pos[:, 1:] - Qtil_pos[:, :-1]) / dt
        Qtil_dxfm = (Qtil_xfm[:, 1:] - Qtil_xfm[:, :-1]) / dt

        dZgnd = (Zgnd[:, 1:] - Zgnd[:, :-1]) / dt

        # Pose-based losses
        loss_lpos = torch.mean(75.0 * torch.abs(Ygnd_pos - Ytil_pos))
        loss_ltxy = torch.mean(10.0 * torch.abs(Ygnd_txy - Ytil_txy))
        loss_lvel = torch.mean(10.0 * torch.abs(Ygnd_vel - Ytil_vel))
        loss_lang = torch.mean(1.25 * torch.abs(Ygnd_ang - Ytil_ang))
        loss_lrvel = torch.mean(2.0 * torch.abs(Ygnd_rvel - Ytil_rvel))
        loss_lrang = torch.mean(2.0 * torch.abs(Ygnd_rang - Ytil_rang))
        loss_lextra = torch.mean(2.0 * torch.abs(Ygnd_extra - Ytil_extra))

        loss_cpos = torch.mean(15.0 * torch.abs(Qgnd_pos - Qtil_pos))
        loss_cxfm = torch.mean(5.0 * torch.abs(Qgnd_xfm - Qtil_xfm))
        loss_cvel = torch.mean(2.0 * torch.abs(Qgnd_vel - Qtil_vel))
        loss_cang = torch.mean(0.75 * torch.abs(Qgnd_ang - Qtil_ang))

        # Velocity losses
        loss_lvel_pos = torch.mean(10.0 * torch.abs(Ygnd_dpos - Ytil_dpos))
        loss_lvel_txy = torch.mean(0.75 * torch.abs(Ygnd_dtxy - Ytil_dtxy))
        loss_cvel_pos = torch.mean(2.0 * torch.abs(Qgnd_dpos - Qtil_dpos))
        loss_cvel_xfm = torch.mean(0.75 * torch.abs(Qgnd_dxfm - Qtil_dxfm))

        # Regularization losses
        loss_sreg = torch.mean(0.1 * torch.abs(Zgnd))
        loss_lreg = torch.mean(0.1 * torch.square(Zgnd))
        loss_vreg = torch.mean(0.01 * torch.abs(dZgnd))

        loss = (
                loss_lpos +
                loss_ltxy +
                loss_lvel +
                loss_lang +
                loss_lrvel +
                loss_lrang +
                loss_lextra +
                loss_cpos +
                loss_cxfm +
                loss_cvel +
                loss_cang +
                loss_lvel_pos +
                loss_lvel_txy +
                loss_cvel_pos +
                loss_cvel_xfm +
                loss_sreg +
                loss_lreg +
                loss_vreg
        )

        # Backpropagation
        loss.backward()

        # c_optimizer.step()
        # d_optimizer.step()
        optimizer.step()

        # Logging
        writer.add_scalars('decompressor/loss', {'loss': loss.item()}, i)
        writer.add_scalars('decompressor/loss_terms', {
            'loc_pos': loss_lpos.item(),
            'loc_txy': loss_ltxy.item(),
            'loc_vel': loss_lvel.item(),
            'loc_ang': loss_lang.item(),
            'loc_rvel': loss_lrvel.item(),
            'loc_rang': loss_lrang.item(),
            'loc_extra': loss_lextra.item(),
            'chr_pos': loss_cpos.item(),
            'chr_xfm': loss_cxfm.item(),
            'chr_vel': loss_cvel.item(),
            'chr_ang': loss_cang.item(),
            'lvel_pos': loss_lvel_pos.item(),
            'lvel_rot': loss_lvel_txy.item(),
            'cvel_pos': loss_cvel_pos.item(),
            'cvel_rot': loss_cvel_xfm.item(),
            'sreg': loss_sreg.item(),
            'lreg': loss_lreg.item(),
            'vreg': loss_vreg.item()
        }, i)
        writer.add_scalars('decompressor/latent', {
            'mean': Zgnd.mean().item(),
            'std': Zgnd.std().item()
        }, i)

        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        if i % 10 == 0:
            # sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, loss))

        if i % 10000 == 0:
            _generate_anim()
            _save_compressed_database()
            save_network('train_ris/decompressor.bin', [
                decompressor.layer1,
                decompressor.predict],
                         decompressor_mean_in,
                         decompressor_std_in,
                         decompressor_mean_out,
                         decompressor_std_out
                         )

        if i % 1000 == 0:
            # c_scheduler.step()
            # d_scheduler.step()
            scheduler.step()
