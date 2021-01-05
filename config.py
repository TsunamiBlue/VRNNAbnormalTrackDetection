"""
NOTICE: THIS CONFIG FILE CONTAINS ALL HARDCODED HYPERPARAMETERS, SOURCE PATH AND ALL OTHER CONFIGS THAT YOU WANT
TO EDIT.

Jan 2021 by Jieli Zheng
anythingbuttusnami@gmail.com

"""
import os

# hyperparameters

x_dim = 28
h_dim = 100
z_dim = 16
n_layers = 1
n_epochs = 4
clip = 10
learning_rate = 1e-3
batch_size = 128
fix_seed = 128
print_every = 100
save_every = 10

# PATH
MAIN_PATH = os.path.abspath(os.path.dirname(__file__) + "./../")
TRAINING_DATA_PATH = f"{MAIN_PATH}/data/trainingData"
MODEL_DATA_PATH = f"{MAIN_PATH}/saves"
