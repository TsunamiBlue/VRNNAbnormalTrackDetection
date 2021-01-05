import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import VRNN
import config as cfg
import pickle
import numpy as np

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def train(epoch):
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
    """uses test data to evaluate
	likelihood of the model"""

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

# use manual seed
torch.manual_seed(seed)
plt.ion()

# handling training & testing data
# TODO
training_data =
training_labels = np.ones(len(training_data))
training_dataset = torch.utils.data.TensorDataset(training_data,training_labels)

# init model + optimizer + datasets
train_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    testing_dataset,
    batch_size=batch_size, shuffle=True)

model = VRNN(x_dim, h_dim, z_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):

    # training + testing
    train(epoch)
    test(epoch)

    # saving model
    if epoch % save_every == 1:
        fn = 'saves/vrnn_state_dict_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to ' + fn)
