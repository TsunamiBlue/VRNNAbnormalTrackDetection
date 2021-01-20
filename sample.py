"""
THIS FILE IS UNNECESSARY AND CAN BE REMOVED.

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from vrnnPytorch import VRNN
import config as cfgs
import os

x_dim = cfgs.x_dim
h_dim = cfgs.h_dim
z_dim = cfgs.z_dim
n_layers = cfgs.n_layers

state_dict = torch.load(os.path.join(cfgs.MODEL_DATA_PATH,"vrnn_state_dict_test.pth"))
current_model = VRNN(x_dim, h_dim, z_dim, n_layers)
current_model.load_state_dict(state_dict)
sample = current_model.sample(28*6)
plt.imshow(sample.numpy(), cmap='gray')
plt.show()
