import torch
import numpy as np
import onnxruntime as ort
import my_modules.NNModels as NNModels
from train_common import load_network, save_network_onnx
from train_common import load_database, load_features, load_latent
import my_modules.quat_functions as quat
import bvh

mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/stepper/stepper.bin')

stepper = NNModels.Stepper.load(mean_in, std_in, mean_out, std_out, layers)
x = torch.ones_like(torch.as_tensor(mean_in.copy()))
save_network_onnx(stepper,
                              x,
                              'train_ris/stepper/stepper.onnx')

mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/decompressor/decompressor.bin')

decom = NNModels.Decompressor.load(mean_in, std_in, mean_out, std_out, layers)
x = torch.ones_like(torch.as_tensor(mean_in.copy()))
save_network_onnx(decom,
                              x,
                              'train_ris/decompressor/decomp.onnx')

