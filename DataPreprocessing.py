import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
# import matplotlib
from matplotlib import pyplot as plt
from vrnnPytorch import VRNN
import config as cfgs
from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import defaultdict


def down_sampling(tracks: list, mmsi, interval=600):
    """
    downsample the tracks to 10 minutes(600 seconds) per point by default.
    :param mmsi: the mmsi of a ship which these track points belong to.
    :param interval: sample interval by second
    :param tracks: track info list with each element: [lat,lon,sog,cog,timestamp]
    :return: modified track info list
    """
    print(f"down sampling by {interval} seconds for mmsi:{mmsi}...")
    ans = [tracks[0]]
    idx = 0
    current_timestamp = tracks[0][-1]
    while idx < len(tracks):
        if tracks[idx][-1] >= current_timestamp + interval:
            ans.append(tracks[idx])
            current_timestamp = tracks[idx][-1]
        idx += 1

    print(f"{len(ans)} track points after downsampling.")
    return ans


def data_preprocessing(raw_data, unused_attribute=None):
    """
    preprocessing track dataset by mmsi.
    raw_data is initially a list of logs with mmsi, after preprocessing there will be no mmsi info to prevent
    overfitting and logs are combined into tracks by mmsi.

    Firstly, process downsampling to avoid four-hot encoding flooding with massive data in a short period of time.

    NOTICE
    downsampling is set to 10 mins by default.
    :param unused_attribute: unnecessary attribute "heading" for four-hot encoding. can introduce others if you want.
    :param raw_data: a numpy ndarray which contains all logs retrieved from ais.
    :return: Tensor:[attribute size, dataset size,time size]
    """
    if unused_attribute is None:
        unused_attribute = [5]
    print("Find tracks by mmsi...")
    track_dict = defaultdict(lambda: [])
    ans = []
    raw_data = np.delete(raw_data, unused_attribute, axis=1)
    for log in raw_data:
        track_dict[log[0]].append(log[1:])

    for k, _ in track_dict.items():
        track_dict[k].sort(key=lambda x: x[4])
        # 1  down sampling and interval set to 600 seconds by default
        track_dict[k] = down_sampling(track_dict[k], k)

    return ans
