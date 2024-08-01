import sys
import struct

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

import my_modules.NNModels as NNModels
import my_modules.quat_functions as quat
import my_modules.xform_functions as xform
from train_common import load_database, load_features, load_latent, save_network, save_network_onnx

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load data
    database = load_database('./data/database.bin')
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    del database

    X = load_features('./data/features.bin')['features'].copy().astype(np.float32)
    Z = load_latent('./train_ris/decompressor/latent.bin')['latent'].copy().astype(np.float32)

    nframes = X.shape[0]
    nfeatures = X.shape[1]
    nlatent = Z.shape[1]

    # Parameters
    seed = 1234
    batchsize = 32
    lr = 0.001
    niter = 500000
    window = 20
    dt = 1.0 / 60.0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    # Fit acceleration structure for nearest neighbors search
    tree = BallTree(X)

    # Means/stds
    X_scale = X.std()
    X_noise_std = X.std(axis=0) + 1.0

    projector_mean_out = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
        Z.mean(axis=0).ravel(),
    ]).astype(np.float32))

    projector_std_out = torch.as_tensor(np.hstack([
        X.std(axis=0).ravel(),
        Z.std(axis=0).ravel(),
    ]).astype(np.float32))

    projector_mean_in = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
    ]).astype(np.float32))

    projector_std_in = torch.as_tensor(np.hstack([
        X_scale.repeat(nfeatures),
    ]).astype(np.float32))

    projector = NNModels.Projector(nfeatures, nfeatures + nlatent)

    def generate_predictions():
        with torch.no_grad():
            # Get slice of database for first clip
            start = range_starts[2]
            stop = min(start + 1000, range_stops[2])

            nsigma = np.random.uniform(size=[stop - start, 1]).astype(np.float32)
            noise = np.random.normal(size=[stop - start, nfeatures]).astype(np.float32)
            Xhat = X[start:stop] + X_noise_std * nsigma * noise

            # Find nearest
            nearest = tree.query(Xhat, k=1, return_distance=False)[:, 0]

            Xgnd = torch.as_tensor(X[nearest])
            Zgnd = torch.as_tensor(Z[nearest])
            Xhat = torch.as_tensor(Xhat)

            # Project
            output = (projector((Xhat - projector_mean_in) / projector_std_in) *
                      projector_std_out + projector_mean_out)

            Xtil = output[:, :nfeatures]
            Ztil = output[:, nfeatures:]

            # Write features
            fmin, fmax = Xhat.cpu().numpy().min(), Xhat.cpu().numpy().max()

            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2 * nfeatures))
            for i in range(nfeatures):
                axs[i].plot(Xgnd[:500:4, i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].plot(Xtil[:500:4, i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].plot(Xhat[:500:4, i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].set_ylim(fmin, fmax)
            plt.tight_layout()

            try:
                plt.savefig('train_ris/projector/projector_X.png')
            except IOError as e:
                print(e)

            plt.close()

            # Write latent
            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()

            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2 * nlatent))
            for i in range(nlatent):
                axs[i].plot(Zgnd[:500:4, i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].plot(Ztil[:500:4, i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].set_ylim(lmin, lmax)
            plt.tight_layout()

            try:
                plt.savefig('train_ris/projector/projector_Z.png')
            except IOError as e:
                print(e)

            plt.close()

    # Train
    writer = SummaryWriter()

    optimizer, scheduler = NNModels.TrainSettings(projector)

    rolling_loss = None

    sys.stdout.write('\n')

    for i in range(niter):
        optimizer.zero_grad()

        # Extract batch
        samples = np.random.randint(0, nframes, size=[batchsize])

        nsigma = np.random.uniform(size=[batchsize, 1]).astype(np.float32)
        noise = np.random.normal(size=[batchsize, nfeatures]).astype(np.float32)
        Xhat = X[samples] + X_noise_std * nsigma * noise

        # Find nearest
        nearest = tree.query(Xhat, k=1, return_distance=False)[:, 0]

        Xgnd = torch.as_tensor(X[nearest])
        Zgnd = torch.as_tensor(Z[nearest])
        Xhat = torch.as_tensor(Xhat)
        Dgnd = torch.sqrt(torch.sum(torch.square(Xhat - Xgnd), dim=-1))

        # Projector
        output = (projector((Xhat - projector_mean_in) / projector_std_in) *
                  projector_std_out + projector_mean_out)

        Xtil = output[:, :nfeatures]
        Ztil = output[:, nfeatures:]
        Dtil = torch.sqrt(torch.sum(torch.square(Xhat - Xtil), dim=-1))

        # Compute Losses
        loss_xval = torch.mean(1.0 * torch.abs(Xgnd - Xtil))
        loss_zval = torch.mean(5.0 * torch.abs(Zgnd - Ztil))
        loss_dist = torch.mean(0.3 * torch.abs(Dgnd - Dtil))
        loss = loss_xval + loss_zval + loss_dist

        # Backprop
        loss.backward()

        optimizer.step()

        # Logging
        writer.add_scalar('projector/loss', loss.item(), i)

        writer.add_scalars('projector/loss_terms', {
            'xval': loss_xval.item(),
            'zval': loss_zval.item(),
            'dist': loss_dist.item(),
        }, i)

        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))

        if i % 10000 == 0:
            generate_predictions()
            save_network('train_ris/projector/projector_1.bin', [
                projector.layer1,
                projector.layer2,
                projector.layer3,
                projector.layer4,
                projector.predict],
                         projector_mean_in,
                         projector_std_in,
                         projector_mean_out,
                         projector_std_out)
            save_network_onnx(projector,
                              projector_mean_in,
                              'train_ris/projector/projector_1.onnx')

        if i % 1000 == 0:
            scheduler.step()
