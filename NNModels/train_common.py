import struct
import torch
import numpy as np
import torch.nn as nn
import my_modules.NNModels as NNModels


def load_database(filename):
    with open(filename, 'rb') as f:

        nframes, nbones = struct.unpack('II', f.read(8))
        bone_positions = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])

        nframes, nbones = struct.unpack('II', f.read(8))
        bone_velocities = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])

        nframes, nbones = struct.unpack('II', f.read(8))
        bone_rotations = np.frombuffer(f.read(nframes*nbones*4*4), dtype=np.float32, count=nframes*nbones*4).reshape([nframes, nbones, 4])

        nframes, nbones = struct.unpack('II', f.read(8))
        bone_angular_velocities = np.frombuffer(f.read(nframes*nbones*3*4), dtype=np.float32, count=nframes*nbones*3).reshape([nframes, nbones, 3])

        nbones = struct.unpack('I', f.read(4))[0]
        bone_parents = np.frombuffer(f.read(nbones * 4), dtype=np.int32, count=nbones).reshape([nbones])

        nranges = struct.unpack('I', f.read(4))[0]
        range_starts = np.frombuffer(f.read(nranges * 4), dtype=np.int32, count=nranges).reshape([nranges])

        nranges = struct.unpack('I', f.read(4))[0]
        range_stops = np.frombuffer(f.read(nranges * 4), dtype=np.int32, count=nranges).reshape([nranges])

        nframes, ncontacts = struct.unpack('II', f.read(8))
        contact_states = np.frombuffer(f.read(nframes * ncontacts), dtype=np.int8, count=nframes * ncontacts).reshape(
            [nframes, ncontacts])

        return {
            'bone_positions': bone_positions,
            'bone_rotations': bone_rotations,
            'bone_velocities': bone_velocities,
            'bone_angular_velocities': bone_angular_velocities,
            'bone_parents': bone_parents,
            'range_starts': range_starts,
            'range_stops': range_stops,
            'contact_states': contact_states,
        }


def load_features(filename):
    with open(filename, 'rb') as f:
        nframes, nfeatures = struct.unpack('II', f.read(8))
        features = np.frombuffer(f.read(nframes * nfeatures * 4), dtype=np.float32, count=nframes * nfeatures).reshape(
            [nframes, nfeatures])

        nfeatures = struct.unpack('I', f.read(4))[0]
        features_offset = np.frombuffer(f.read(nfeatures * 4), dtype=np.float32, count=nfeatures).reshape([nfeatures])

        nfeatures = struct.unpack('I', f.read(4))[0]
        features_scale = np.frombuffer(f.read(nfeatures * 4), dtype=np.float32, count=nfeatures).reshape([nfeatures])

    return {
        'features': features,
        'features_offset': features_offset,
        'features_scale': features_scale,
    }


def load_latent(filename):
    with open(filename, 'rb') as f:
        nframes, nfeatures = struct.unpack('II', f.read(8))
        latent = np.frombuffer(f.read(nframes * nfeatures * 4), dtype=np.float32, count=nframes * nfeatures).reshape(
            [nframes, nfeatures])

    return {
        'latent': latent,
    }


def save_network(filename, layers, mean_in, std_in, mean_out, std_out):
    with torch.no_grad():
        with open(filename, 'wb') as f:
            f.write(struct.pack('I', mean_in.shape[0]) + mean_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', std_in.shape[0]) + std_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', mean_out.shape[0]) + mean_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', std_out.shape[0]) + std_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', len(layers)))
            for layer in layers:
                f.write(struct.pack('II', *layer.weight.T.shape) + layer.weight.T.cpu().numpy().astype(
                    np.float32).ravel().tobytes())
                f.write(
                    struct.pack('I', *layer.bias.shape) + layer.bias.cpu().numpy().astype(np.float32).ravel().tobytes())


def load_network(filename):
    with open(filename, 'rb') as f:
        mean_in_len = struct.unpack('I', f.read(4))[0]
        mean_in = np.frombuffer(f.read(mean_in_len * 4), dtype=np.float32)

        std_in_len = struct.unpack('I', f.read(4))[0]
        std_in = np.frombuffer(f.read(std_in_len * 4), dtype=np.float32)

        mean_out_len = struct.unpack('I', f.read(4))[0]
        mean_out = np.frombuffer(f.read(mean_out_len * 4), dtype=np.float32)

        std_out_len = struct.unpack('I', f.read(4))[0]
        std_out = np.frombuffer(f.read(std_out_len * 4), dtype=np.float32)

        num_layers = struct.unpack('I', f.read(4))[0]

        layers = []
        for _ in range(num_layers):
            weight_shape = struct.unpack('II', f.read(8))
            weight_size = weight_shape[0] * weight_shape[1]
            weight = np.frombuffer(f.read(weight_size * 4), dtype=np.float32).reshape(weight_shape).T.copy()

            bias_shape = struct.unpack('I', f.read(4))[0]
            bias = np.frombuffer(f.read(bias_shape * 4), dtype=np.float32).copy()

            layer = nn.Linear(weight_shape[1], weight_shape[0])  # Transpose back the shape
            layer.weight.data = torch.from_numpy(weight)
            layer.bias.data = torch.from_numpy(bias)

            layers.append(layer)

    return mean_in, std_in, mean_out, std_out, layers


def save_network_onnx(model, input_size, filename):
    # Example tensor
    x = torch.ones_like(input_size)

    # Export the model
    torch.onnx.export(model,
                      x,
                      filename,
                      export_params=True,
                      opset_version=9,
                      do_constant_folding=True,
                      input_names=['X'],
                      output_names=['Y']
                      )



