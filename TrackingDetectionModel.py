import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from vrnnPytorch import VRNN
import config as cfg
from sklearn.model_selection import train_test_split
import numpy as np

"""
    A tracking detection model using VRNN as the core algorithm based on Pytorch.
    Jan 2021 by Jieli Zheng
    anythingbuttusnami@gmail.com
"""


class TrackingDetectionModel:
    """
    A deep learning class class all in one.
    """
    def __init__(self, cfg, raw):
        """
         init service with all hyper-params
        :param cfg: see config.py
        :param raw: raw data
        """
        # set hyperparameters
        self.raw_data = raw
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

        print("Generating Dataset...")
        # handling training & testing data
        (training_data, test_data) = train_test_split(raw_data, split_ratio)
        # training_labels = np.ones(len(training_data))
        training_dataset = torch.utils.data.TensorDataset(training_data)
        testing_dataset = torch.utils.data.TensorDataset(test_data)
        # generate loader
        self.train_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            testing_dataset,
            batch_size=batch_size, shuffle=True)

        print("initializing vrnn model...")
        # init vrnn models
        self.model = VRNN(x_dim, h_dim, z_dim, n_layers)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("DONE.")

    def set_fixed_seed(self,seed):
        """
        always set fixed seed before training if consistent performance is expected.
        :param seed: Float
        :return: None
        """
        self.seed = seed

    def plot_activate(self):
        # activate plotting
        plt.ion()

    def train_from_scratch(self):
        """
        zero-knowledge in this area and train model from scratch with training set.
        :return:
        """
        for epoch in range(1, n_epochs + 1):

            self.train_one_epoch(epoch)
            # TODO re
            test(i)

            # saving model
            if epoch % save_every == 1:
                fn = 'saves/vrnn_state_dict_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), fn)
                print('Saved model to ' + fn)

    def train_one_epoch(self,epoch):

        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):

            # transforming data
            # data = Variable(data)
            # to remove eventually
            data = Variable(data.squeeze().transpose(0, 1))
            data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())

            # forward, backward, optimize
            optimizer.zero_grad()
            kld_loss, nll_loss, _, _ = model(data)
            loss = kld_loss + nll_loss
            loss.backward()
            optimizer.step()

            # grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm(model.parameters(), clip)

            # printing
            if batch_idx % print_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           kld_loss.data.item() / batch_size,
                           nll_loss.data.item() / batch_size))

                sample = model.sample(28)
                plt.imshow(sample.numpy())
                plt.pause(1e-6)

            train_loss += loss.data.item()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    def validate_current_epoch(selfepoch):


def train(epoch, optimizer):
    """
    Training based on Variation Recurrent Neural Network
    :param optimizer: any torch optimizer
    :param epoch: Int
    :return: None
    """
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        # transforming data
        # data = Variable(data)
        # to remove eventually
        data = Variable(data.squeeze().transpose(0, 1))
        data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        # printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       kld_loss.data.item() / batch_size,
                       nll_loss.data.item() / batch_size))

            sample = model.sample(28)
            plt.imshow(sample.numpy())
            plt.pause(1e-6)

        train_loss += loss.data.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    """
     Evaluation based on both KL-Divergence and Negative Log Likelihood loss
    :param epoch: Int
    :return:
    """
    mean_kld_loss, mean_nll_loss = 0, 0
    for i, (data, _) in enumerate(test_loader):
        # data = Variable(data)
        data = Variable(data.squeeze().transpose(0, 1))
        data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())

        kld_loss, nll_loss, _, _ = model(data)
        mean_kld_loss += kld_loss.data.item()
        mean_nll_loss += nll_loss.data.item()

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)

    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))


if __name__ == '__main__':
    # hyperparameters
    x_dim = cfg.x_dim
    h_dim = cfg.h_dim
    z_dim = cfg.z_dim
    n_layers = cfg.n_layers
    n_epochs = cfg.n_epochs
    clip = cfg.clip
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    seed = cfg.fix_seed
    print_every = cfg.print_every
    save_every = cfg.save_every
    split_ratio = cfg.split_ratio

    # use manual seed
    torch.manual_seed(seed)
    plt.ion()

    # handling training & testing data
    # TODO it's a placeholder for raw data.
    raw_data = []
    (training_data, test_data) = train_test_split(raw_data, test_size=split_ratio,random_state=42)
    # training_labels = np.ones(len(training_data))
    training_dataset = torch.utils.data.TensorDataset(training_data)
    testing_dataset = torch.utils.data.TensorDataset(test_data)
    # generate loader
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=batch_size, shuffle=True)

    # init vrnn models
    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):

        train(epoch, optimizer=optimizer)
        test(epoch)

        # saving model
        if epoch % save_every == 1:
            fn = 'saves/vrnn_state_dict_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to ' + fn)
