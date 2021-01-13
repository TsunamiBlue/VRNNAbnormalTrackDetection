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

def down_sampling(tracks:list,interval = 600):
    """
    downsample the tracks to 10 minutes(600 seconds) per point by default.
    :param interval: sample interval by second
    :param tracks: track info list with each element: [lat,lon,sog,cog,timestamp]
    :return: modified track info list
    """
    print(f"down sampling by {interval} seconds...")
    ans = []
    idx = 0
    current_timestamp = tracks[0][]
    while idx < tracks.__len__():
        if tracks[-1]

    return ans


def data_preprocessing(raw_data):
    """
    preprocessing track dataset by mmsi.
    raw_data is initially a list of logs with mmsi, after preprocessing there will be no mmsi info to prevent
    overfitting and logs are combined into tracks by mmsi.
    NOTICE
    downsampling is set to 10 mins.
    :param raw_data: a numpy ndarray which contains all logs retrieved from ais.
    :return: Tensor:[attribute size, dataset size,time size]
    """
    print("Find tracks by mmsi...")
    track_dict = defaultdict(lambda: [])
    ans = []
    for log in raw_data:
        track_dict[log[0]].append(log[1:])

    for k, _ in track_dict.items():
        track_dict[k].sort(key=lambda x: x[5])
        # down sampling and interval set to 600 seconds by default
        track_dict[k] = down_sampling(track_dict[k])
        print(track_dict[k].__len__())

    for


    return ans
