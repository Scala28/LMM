import torch
import numpy as np
from train_common import load_network, save_network_onnx
import my_modules.NNModels as NNModels

mean_in, std_in, mean_out, std_out, layers = load_network('train_ris/stepper/stepper.bin')

model = NNModels.Stepper.load(mean_in, std_in, mean_out, std_out, layers)

print(mean_in.shape)

save_network_onnx(model, torch.as_tensor(mean_in.copy()), 'train_ris/stepper/stepper.onnx')


