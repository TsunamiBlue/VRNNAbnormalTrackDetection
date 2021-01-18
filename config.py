"""
NOTICE: THIS CONFIG FILE CONTAINS ALL HARDCODED HYPERPARAMETERS, SOURCE PATH AND ALL OTHER CONFIGS THAT YOU WANT
TO EDIT.

Jan 2021 by Jieli Zheng
anythingbuttusnami@gmail.com

"""
import os

# FLAG BETWEEN MNIST OR AIS
TEST_FLAG = False

# hyperparameters

x_dim = 4
h_dim = 100
z_dim = 16
n_layers = 1
n_epochs = 10
clip = 10
learning_rate = 1e-3
batch_size = 4
fix_seed = 128
print_every = 100
save_every = 10
split_ratio = 0.2
cross_validation = False

# analysis boundaries
# coordinate, unit is degree
LAT_MIN = -90
LAT_MAX = 90.0
LON_MIN = -180
LON_MAX = 180
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
# speed, unit is knots
SOG_MIN = 1
SOG_MAX = 30.0
# course, unit is degree
COG_MIN = 0
COG_MAX = 360
# heading, unit is degree
HEADING_MIN = 0
HEADING_MAX = 0

# PATH
MAIN_PATH = os.path.abspath(os.path.dirname(__file__) + "./../")
TRAINING_DATA_PATH = os.path.join(MAIN_PATH,'VRNNAbnormalTrackingDetection','data','trainingData')
# print(TRAINING_DATA_PATH)
MODEL_DATA_PATH = os.path.join(MAIN_PATH,'VRNNAbnormalTrackingDetection','data','saves')
