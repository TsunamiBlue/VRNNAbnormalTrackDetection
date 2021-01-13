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
from DataPreprocessing import data_preprocessing

"""
    A tracking detection model using VRNN as the core algorithm based on Pytorch.
    Jan 2021 by Jieli Zheng
    anythingbuttusnami@gmail.com
"""


class TrackingDetectionModel:
    """
    A deep learning class, all in one.
    """

    def __init__(self, cfg):
        """
         init service with all hyper-params
        :param cfg: see config.py
        :param raw: raw data
        """
        # set hyperparameters
        self.x_dim = cfg.x_dim
        self.h_dim = cfg.h_dim
        self.z_dim = cfg.z_dim
        self.n_layers = cfg.n_layers
        self.n_epochs = cfg.n_epochs
        self.clip = cfg.clip
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.print_every = cfg.print_every
        self.save_every = cfg.save_every
        self.split_ratio = cfg.split_ratio
        self.cross_validation = cfg.cross_validation
        self.seed = None
        # please transmit data before training!
        self.raw_data = None
        self.train_loader = None
        self.test_loader = None

        print("initializing vrnn model...")
        # init vrnn models
        self.model = VRNN(self.x_dim, self.h_dim, self.z_dim, self.n_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print("DONE.")

    def generate_dataloader(self, raw, already_loaded=False):
        """
        :param:raw: A sequence of data for model, or with already_loaded True, input a tuple of two already generated
                    Torch.utils.data.TensorDataset for training and testing set in order.
        :return: None
        """
        self.raw_data = raw
        print("Generating Dataset...")
        # handling training & testing data
        if already_loaded:
            self.train_loader = raw[0]
            self.test_loader = raw[1]
        else:
            (training_data, test_data) = train_test_split(self.raw_data, test_size=self.split_ratio)
            # training_labels = np.ones(len(training_data))
            training_dataset = AISDataset(training_data)
            testing_dataset = AISDataset(test_data)
            # generate loader
            self.train_loader = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=self.batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                testing_dataset,
                batch_size=self.batch_size, shuffle=True)

    def set_fixed_seed(self, seed):
        """
        always set fixed seed before training if consistent performance is expected.
        :param seed: Float
        :return: None
        """
        self.seed = seed

    def plot_activate(self):
        # activate plotting
        plt.ion()

    def train_from_scratch(self, output_path=None):
        """
        zero-knowledge in this area and train model from scratch with training set.
        :param: n_epochs:int, you can change the number of epochs before training.
        :return:
        """
        for epoch in range(1, self.n_epochs + 1):
            self.train_one_epoch(epoch)
            self.validate_current_epoch(epoch)

        # save model to given path
        if output_path is not None:
            path = os.path.join(output_path, 'vrnn_state_dict_test.pth')
            torch.save(self.model.state_dict(), path)
            print(f"DONE. Saved model at {path}")

    def train_one_epoch(self, epoch):
        """
        Training based on Variation Recurrent Neural Network
        :param epoch: int
        :return: None
        """
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            # print(data)
            # transforming data
            # data = Variable(data)
            # to remove eventually
            data = Variable(data[0].squeeze())
            print(data.shape)
            data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())
            print(data.shape)
            # forward, backward, optimize
            self.optimizer.zero_grad()
            kld_loss, nll_loss, enc_log, dec_log = self.model(data)
            loss = kld_loss + nll_loss
            loss.backward()

            # grad norm clipping, only in pytorch version >= 1.10
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            # printing
            if batch_idx % self.print_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           kld_loss.data.item() / self.batch_size,
                           nll_loss.data.item() / self.batch_size))

                sample = self.model.sample(28)
                plt.imshow(sample.numpy())
                plt.pause(1e-6)

            train_loss += loss.data.item()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def validate_current_epoch(self, epoch):
        """
        Evaluation based on both KL-Divergence and Negative Log Likelihood loss
        :param epoch: Int
        :return:
        """
        mean_kld_loss, mean_nll_loss = 0, 0
        for i, (data, _) in enumerate(self.test_loader):
            # data = Variable(data)
            data = Variable(data.squeeze().transpose(0, 1))
            data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())

            kld_loss, nll_loss, _, _ = self.model(data)
            mean_kld_loss += kld_loss.data.item()
            mean_nll_loss += nll_loss.data.item()

        mean_kld_loss /= len(self.test_loader.dataset)
        mean_nll_loss /= len(self.test_loader.dataset)

        print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
            mean_kld_loss, mean_nll_loss))


class AISDataset(torch.utils.data.Dataset):
    """
    a dataset class for ais data.
    """

    def __init__(self, datapoints):
        super().__init__()
        self.raw_data = datapoints
        self.shape = datapoints.shape

    def __getitem__(self, idx):
        return self.raw_data[idx]

    def __len__(self):
        return len(self.raw_data)





if __name__ == '__main__':
    plt.ion()
    flag = False
    # flag = True
    # test model with mnist
    if flag:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=cfgs.batch_size, shuffle=True)
        print(datasets.MNIST('data', train=False,
                             transform=transforms.ToTensor()).data.shape)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False,
                           transform=transforms.ToTensor()),
            batch_size=cfgs.batch_size, shuffle=True)
        mnist_data = (train_loader, test_loader)
        TDModel = TrackingDetectionModel(cfgs)
        TDModel.generate_dataloader(raw=mnist_data, already_loaded=True)
        TDModel.train_from_scratch(output_path=cfgs.MODEL_DATA_PATH)
    else:
        # ty w/ ais dataset
        ais_data = np.loadtxt(os.path.join(cfgs.TRAINING_DATA_PATH, '202012.txt'), delimiter=',')
        ais_data = data_preprocessing(ais_data)
        ais_dataset = AISDataset(ais_data)
        TDModel = TrackingDetectionModel(cfgs)
        TDModel.generate_dataloader(raw=ais_dataset)
        # print(TDModel.train_loader.dataset.shape)
        # TDModel.train_from_scratch(output_path=cfgs.MODEL_DATA_PATH)

    # predictions = [TDModel.model(trans(data)) for i,(data,_) in enumerate(test_loader)]
    # print(predictions)
