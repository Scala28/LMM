import torch
import numpy as np
import my_modules.NNModels as NNModels
from train_common import load_network, save_network_onnx

mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/decompressor/decompressor.bin')
print(mean_in.shape)
print(mean_out.shape)
# decompressor = NNModels.Decompressor.load(mean_in, std_in, mean_out, std_out, layers)
# save_network_onnx(decompressor, torch.as_tensor(mean_in.copy()), 'train_ris/decompressor/decompressor.onnx')
