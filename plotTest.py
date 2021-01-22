import torch
import torch.nn as nn
from vrnnPytorch import VRNN
import config as cfgs
import os
import numpy as np
from TrackingDetectionModel import AISDataset
from TrackingDetectionModel import TrackingDetectionModel
from matplotlib import pyplot as plt
from DataPreprocessing import data_preprocessing
"""
plot and save for existing track dicts.
"""
#
# def plot_track(track_dict, abnormal_dict=None):
# 	"""
# 	:param track_dict: a dictionary for normal tracks, mmsi->[lat,lon,sog,cog]
# 	:param abnormal_dict: a dictionary for abnormal tracks, mmsi->[lat,lon,sog,cog]
# 	"""
#
#
#
#
#
#
#
# def save_pic(path):
ais_data = np.loadtxt(os.path.join(cfgs.TRAINING_DATA_PATH, 'data202012.txt'), delimiter=',')
ais_data = data_preprocessing(ais_data)
ais_dataset = AISDataset(ais_data)
TDModel = TrackingDetectionModel(cfgs)
TDModel.generate_dataloader(raw=ais_dataset)
# print(TDModel.train_loader.dataset.shape)
TDModel.plot_track(ais_data,None)