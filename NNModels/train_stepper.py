import sys
import struct

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

import my_modules.NNModels as NNModels
import my_modules.quat_functions as quat
import my_modules.xform_functions as xform
from train_common import load_database, load_features, load_latent, save_network, save_network_onnx

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

    # Compute means/stds

    X_scale = X.std()
    Z_scale = Z.std()

    stepper_mean_out = torch.as_tensor(np.hstack([
        ((X[1:] - X[:-1]) / dt).mean(axis=0).ravel(),
        ((Z[1:] - Z[:-1]) / dt).mean(axis=0).ravel(),
    ]).astype(np.float32))

    stepper_std_out = torch.as_tensor(np.hstack([
        ((X[1:] - X[:-1]) / dt).std(axis=0).ravel(),
        ((Z[1:] - Z[:-1]) / dt).std(axis=0).ravel(),
    ]).astype(np.float32))

    stepper_mean_in = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
        Z.mean(axis=0).ravel(),
    ]).astype(np.float32))

    stepper_std_in = torch.as_tensor(np.hstack([
        X_scale.repeat(nfeatures),
        Z_scale.repeat(nlatent),
    ]).astype(np.float32))

    X = torch.as_tensor(X)
    Z = torch.as_tensor(Z)

    # NN Model
    stepper = NNModels.Stepper(nfeatures + nlatent)

    def generate_predictions():
        with torch.no_grad():
            # Get a clip

            start = range_starts[2]
            stop = min(start + 1000, range_stops[2])

            Xgnd = X[start:stop][np.newaxis]
            Zgnd = Z[start:stop][np.newaxis]

            Xtil = Xgnd.clone()
            Ztil = Zgnd.clone()

            for k in range(1, start - stop):
                if (k-1) % window == 0:  # Simulating the Projector's goal
                    Xtil_prev = Xgnd[:, k-1]
                    Ztil_prev = Zgnd[:, k-1]
                else:
                    Xtil_prev = Xtil[:, k-1]
                    Ztil_prev = Ztil[:, k-1]

                delta = (stepper((torch.cat([Xtil_prev, Ztil_prev], dim=-1) -
                                  stepper_mean_in) / stepper_std_in) *
                         stepper_std_out + stepper_mean_out)
                Xtil[:, k] = Xtil_prev + dt * delta[:, :nfeatures]
                Ztil[:, k] = Ztil_prev + dt * delta[:, nfeatures:]

            # Write features
            fmin, fmax = Xgnd.cpu().numpy().min(), Xgnd.cpu().numpy().max()
            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2 * nfeatures))
            for j in range(nfeatures):
                axs[j].plot(Xgnd[0, :500, j].cpu().numpy())
                axs[j].plot(Xtil[0, :500, j].cpu().numpy())
                axs[j].set_ylim(fmin, fmax)
            plt.tight_layout()
            try:
                plt.savefig('train_ris/stepper/stepper_X.png')
            except IOError as e:
                print(e)

            plt.close()

            # Write latent

            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()

            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2 * nlatent))
            for j in range(nlatent):
                axs[j].plot(Zgnd[0, :500, j].cpu().numpy())
                axs[j].plot(Ztil[0, :500, j].cpu().numpy())
                axs[j].set_ylim(lmin, lmax)
                plt.tight_layout()

            try:
                plt.savefig('train_ris/stepper/stepper_Z.png')
            except IOError as e:
                print(e)

            plt.close()


    # Build batches respecting window size
    indices = []
    for i in range(nframes - window + 1):
        indices.append(np.arange(i, i + window))
    indices = torch.as_tensor(np.array(indices), dtype=torch.long)
    writer = SummaryWriter()

    optimizer, scheduler = NNModels.TrainSettings(stepper)

    rolling_loss = None

    # Train
    for i in range(niter):

        optimizer.zero_grad()

        # Extract batch
        batch = indices[torch.randint(0, len(indices), size=[batchsize])]  # (batchsize, window)

        Xgnd = X[batch]  # (batchsize, window, nfeatures)
        Zgnd = Z[batch]  # (batchsize, window, nlatent)

        Xtil = [Xgnd[:, 0]]  # [(batchsize, nfeatures)]
        Ztil = [Zgnd[:, 0]]

        for _ in range(1, window):
            delta = (stepper((torch.cat([Xtil[-1], Ztil[-1]], dim=-1)
                              - stepper_mean_in) / stepper_std_in) *
                     stepper_std_out + stepper_mean_out)

            Xtil.append(Xtil[-1] + dt * delta[:, :nfeatures])
            Ztil.append(Ztil[-1] + dt * delta[:, nfeatures:])

        Xtil = torch.cat([x[:, np.newaxis] for x in Xtil], dim=1)
        Ztil = torch.cat([z[:, np.newaxis] for z in Ztil], dim=1)

        # Compute velocities
        dXgnd = (Xgnd[:, 1:] - Xgnd[:, :-1]) / dt
        dZgnd = (Zgnd[:, 1:] - Zgnd[:, :-1]) / dt

        dXtil = (Xtil[:, 1:] - Xtil[:, :-1]) / dt
        dZtil = (Ztil[:, 1:] - Ztil[:, :-1]) / dt

        # Compute value Losses
        loss_xval = torch.mean(2.0 * torch.abs(Xgnd - Xtil))
        loss_zval = torch.mean(7.5 * torch.abs(Zgnd - Ztil))

        # Compute velocity losses
        loss_xvel = torch.mean(0.2 * torch.abs(dXgnd - dXtil))
        loss_zvel = torch.mean(0.7 * torch.abs(dZgnd - dZtil))

        loss = loss_xval + loss_xvel + loss_zval + loss_zvel

        # Backpropagation
        loss.backward()

        optimizer.step()

        # Logging
        writer.add_scalars('stepper/loss', {'loss': loss.item()}, i)
        writer.add_scalars('stepper/loss_terms', {
            'x_value': loss_xval.item(),
            'x_velocity': loss_xvel.item(),
            'z_value': loss_zval.item(),
            'z_velocity': loss_zvel.item()
        }, i)

        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))

        if i % 10000 == 0:
            generate_predictions()
            save_network('train_ris/stepper/stepper.bin', [
                stepper.layer1,
                stepper.layer2,
                stepper.predict],
                         stepper_mean_in,
                         stepper_std_in,
                         stepper_mean_out,
                         stepper_std_out)
            save_network_onnx(stepper,
                              stepper_mean_in,
                              'train_ris/stepper/stepper.onnx')

        if i % 1000 == 0:
            scheduler.step()
