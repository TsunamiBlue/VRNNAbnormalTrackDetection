"""
THIS FILE IS UNNECESSARY AND CAN BE REMOVED.

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from vrnnPytorch import VRNN
import config as cfgs
import os
import numpy as np

if __name__ == '__main__':
    # plt.ion()
    test_model_path = os.path.join(cfgs.MODEL_DATA_PATH, "vrnn_state_dict_test.pth")
    test_data_path = os.path.join(cfgs.TRAINING_DATA_PATH, 'data202012.txt')
    your_dirty_dataset = []
    X = []
    Y = []
    const = 1
    with open(test_data_path, 'r') as f:
        for i, data in enumerate(f.readlines()):
            split_data = data.split(',')
            # print(split_data)
            if len(split_data) != 7:
                continue
            x = float(split_data[1]) / const
            y = float(split_data[2]) / const
            modified_data = np.array([x, y])
            # print(modified_data)
            your_dirty_dataset.append(split_data)
            X.append(x)
            Y.append(y)
    f.close()
    print("stop read files...")
    print(f"totally {len(your_dirty_dataset)} points.")
    # print(your_dirty_dataset[:][0])
    # print(X)
    # X = np.random.normal(5.0, 1.0, 3)
    # Y = np.random.normal(5.0, 1.0, 3)
    plt.scatter(X, Y)
    plt.show()
    input()
    print("plot finished...")
    plt.pause(1e6)
