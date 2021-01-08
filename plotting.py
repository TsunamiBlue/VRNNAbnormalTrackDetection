import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from vrnnPytorch import VRNN
import config as cfgs
import os
import numpy as np

if __name__ == '__main__':
    plt.ion()
    test_model_path = os.path.join(cfgs.MODEL_DATA_PATH, "vrnn_state_dict_test.pth")
    test_data_path = os.path.join(cfgs.TRAINING_DATA_PATH, 'data1.txt')
    your_dirty_dataset = []
    X = []
    Y = []
    const = 1
    with open(test_data_path, 'r') as f:
        for i, data in enumerate(f.readlines()):
            x = float(data.split(',')[1]) / const
            y = float(data.split(',')[2]) / const
            modified_data = np.array([x, y])
            # print(modified_data)
            your_dirty_dataset.append(modified_data)
            X.append(x)
            Y.append(y)
    f.close()
    print("stop read files...")
    # print(your_dirty_dataset[:][0])
    # print(X)
    plt.scatter(X, Y)
    print("plot finished...")
    plt.pause(1e-6)
