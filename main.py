from TrackingDetectionModel import AISDataset
from TrackingDetectionModel import TrackingDetectionModel
from matplotlib import pyplot as plt
import config as cfgs
import os
import numpy as np
from DataPreprocessing import data_preprocessing


"""
It's a local main method to instruct how to use this model.
"""
if __name__ == '__main__':
    # uncomment it to visualize attention
    # plt.ion()
    # ty w/ ais dataset
    ais_data = np.loadtxt(os.path.join(cfgs.TRAINING_DATA_PATH, 'data202012.txt'), delimiter=',')
    ais_data = data_preprocessing(ais_data)
    ais_dataset = AISDataset(ais_data)
    TDModel = TrackingDetectionModel(cfgs)
    TDModel.generate_dataloader(raw=ais_dataset)
    # print(TDModel.train_loader.dataset.shape)
    TDModel.train_from_scratch(output_path=cfgs.MODEL_DATA_PATH)
    # TDModel.plot_track()
    # smp = TDModel.sample_track(287)
    # print(ais_data[0].size())
    TDModel.abnormal_detection(ais_data[0])
