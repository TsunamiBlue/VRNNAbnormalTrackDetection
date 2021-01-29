import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
from vrnnPytorch import VRNN
import config as cfgs
from sklearn.model_selection import train_test_split
import os
import numpy as np
from scipy import stats
import warnings

"""
    A tracking detection model using VRNN as the core algorithm based on Pytorch.
    Jan 2021 by Jieli Zheng
    anythingbuttusnami@gmail.com
"""


class AISDataset(torch.utils.data.Dataset):
    """
    a toy dataset class for ais data.
    """

    def __init__(self, datapoints):
        super().__init__()
        self.data = datapoints
        self.shape = datapoints.shape

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TrackingDetectionModel:
    """
    A deep learning class, all in one.
    """

    def __init__(self, cfg, seq_len=287, abnormal=None):
        """
         init service with all hyper-params
        :param seq_len: single data sequence length after preprocessing
        :param cfg: see config.py
        :param abnormal: can specify abnormal tracks from outer resources.
        """
        # set hyperparameters
        if abnormal is None:
            abnormal = []
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
        self.abnormal = abnormal
        self.train_loader = None
        self.test_loader = None
        # sample granularity & downsample size
        self.sample_granularity = cfg.downsample_interval
        self.single_track_len = None
        # boundary info
        self.lat_max = cfg.LAT_MAX
        self.lat_min = cfg.LAT_MIN
        self.lat_range = cfg.LAT_MAX-cfg.LAT_MIN
        self.lon_max = cfg.LON_MAX
        self.lon_min = cfg.LON_MIN
        self.lon_range = cfg.LON_MAX - cfg.LON_MIN

        print("initializing vrnn model...")
        # init vrnn models
        self.model = VRNN(self.x_dim, self.h_dim, self.z_dim, self.n_layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def generate_dataloader(self, raw: AISDataset, already_loaded=False):
        """
        :param:raw: AISDataset for model, or with already_loaded True, input a tuple of two already generated
                    Torch.utils.data.TensorDataset for training and testing set in order.
        :return: None
        """
        self.raw_data = raw
        print("Generating Dataset...")
        # handling training & testing data
        if already_loaded:
            self.train_loader = raw[0]
            self.test_loader = raw[1]
            self.single_track_len = self.train_loader.dataset[0].size()[0]
        else:
            (training_data, test_data) = train_test_split(self.raw_data, test_size=self.split_ratio)
            self.single_track_len = training_data.size()[1]
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
        # activate plotting while training
        plt.ion()

    def train_from_scratch(self, output_path=None, inner_invoking=False):
        """
        zero-knowledge in this area and train model from scratch with training set.
        NOTICE: all configs for training should be specified in config.py
        :param: output_path: path str
        :param: inner_invoking: invoked from train_after_backbone.
        :return: None
        """
        if output_path is None:
            warnings.warn("Warning: output path not specified.")
            return
        if not inner_invoking:
            print("Start training from scratch...")
        for epoch in range(1, self.n_epochs + 1):
            self._train_one_epoch(epoch)
            self._validate_current_epoch(epoch)

        # save model to given path
        torch.save(self.model.state_dict(), output_path)
        # print(self.model.state_dict().keys())
        if not inner_invoking:
            print(f"DONE. Saved model at {output_path}")
        else:
            print(f"DONE. Updated model at {output_path}")

    def train_after_backbone(self, saved_model_path=None, output_path=None):
        """
        Update the model with pretrained params.
        NOTICE: all configs should be specified in config.py
        :param saved_model_path: path str
        :param output_path: path str
        :return: None
        """
        if saved_model_path is None:
            warnings.warn('Warning: saved model path not specified')
            return

        if output_path is None:
            warnings.warn("Warning: output path not specified")
            return
        print("Start train with specified pretrained model...")
        loaded_params = torch.load(saved_model_path)
        self.model.load_state_dict(loaded_params)

        self.train_from_scratch(output_path=output_path)

    def _train_one_epoch(self, epoch, plot_sample = False):
        """
        Training based on Variation Recurrent Neural Network
        :param plot_sample: True if model attention picture is needed.
        :param epoch: int
        :return: None
        """
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            # forward, backward, optimize
            self.optimizer.zero_grad()
            kld_loss, nll_loss, _ = self.model(data)
            # print(list(self.model.children())[0])
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
                if plot_sample:
                    # can be changed
                    sample = self.model.sample(10)
                    plt.imshow(sample.numpy())
                    plt.pause(1e-6)

            train_loss += loss.data.item()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def _validate_current_epoch(self, epoch):
        """
        Evaluation based on both KL-Divergence and Negative Log Likelihood loss
        :param epoch: Int
        :return: None
        """
        mean_kld_loss, mean_nll_loss = 0, 0
        for batch_idx, data in enumerate(self.test_loader):
            # validating
            kld_loss, nll_loss, _ = self.model(data)
            mean_kld_loss += kld_loss.data.item()
            mean_nll_loss += nll_loss.data.item()

        mean_kld_loss /= len(self.test_loader.dataset)
        mean_nll_loss /= len(self.test_loader.dataset)

        print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
            mean_kld_loss, mean_nll_loss))

    def sample_track(self, track_num):
        """
        sample and reconstruct tracks
        :param track_num: num of track points to reconstruct by default sample granularity.
        :return: reconstructed tracks:Tensor[track_num, sample granularity, attribute]
        """
        if track_num % self.single_track_len != 0:
            warnings.warn("Warning: partial broken tracks generated.")

        return self.model.sample(track_num)

    def abnormal_detection(self,threshold:float, test_data: torch.Tensor):
        """
        abnormal detection method
        :param threshold: tolerance threshold of abnormal tracks
        :param test_data: track Tensor
        :return: True if it's an abnormal track
        """

        saved_model_path = cfgs.MODEL_DATA_PATH
        loaded_params = torch.load(saved_model_path)
        self.model.load_state_dict(loaded_params)

        # use existing model to resample standard data
        sample_data = self.sample_track(test_data.size()[0])
        print(test_data.size()[0])
        sample_data = sample_data.unsqueeze(0)
        kld_loss, nll_loss, pkg = self.model(sample_data)
        standard_ref_enc_mean = pkg[0]
        standard_ref_enc_std = pkg[1]
        # standard_ref_dec_mean = pkg[2]
        # standard_ref_dec_std = pkg[3]

        test_data = test_data.unsqueeze(0)
        kld_loss, nll_loss, pkg = self.model(test_data)
        test_enc_mean, test_enc_std, test_dec_mean, test_dec_std, _, _ = pkg
        # print(f" standard_ref_mean {standard_ref_enc_mean[0].size()} \t standard_ref_std {standard_ref_enc_std[0].size()}")
        # print(f" test_mean {test_enc_mean[0].size()} \t test_std {test_enc_std[0].size()}")
        print()
        ans_flag = self._compare_sequence((standard_ref_enc_mean[0],standard_ref_enc_std[0]), (test_enc_mean[0], test_enc_std[0]),threshold=threshold)

        if not ans_flag:
            self.abnormal.append(test_data.squeeze(0))
            return True
        else:
            return False


    def plot_track(self, normal_tracks=None, abnormal_tracks=None):
        """
        # TODO re-write plot method. data should come from model class & deal with data transferring
        scratch plotting method, should be polished later.
        :param abnormal_tracks: if None will not output abnormal tracks
        :param normal_tracks: if None use model data, [number of tracks, sample points, attribute]
	    """
        cmap = plt.cm.get_cmap("Blues")
        normal_size = len(normal_tracks)
        print(normal_size)
        for idx, track_tensor in enumerate(normal_tracks):
            c = cmap(float(idx)/(normal_size-1))
            v_lat = track_tensor[:, 0] * self.lat_range + self.lat_min
            v_lon = track_tensor[:, 1] * self.lon_range + self.lon_min
            plt.plot(v_lon, v_lat, color=c, linewidth=0.8)
        cmap2 = plt.cm.get_cmap("autumn")
        if abnormal_tracks is not None:
            abnormal_size = len(abnormal_tracks)
            print(abnormal_size)
            for idx, track_tensor in enumerate(abnormal_tracks):
                c2 = cmap2(float(idx)/(abnormal_size-1))
                v_lat = track_tensor[:, 0] * self.lat_range + self.lat_min
                v_lon = track_tensor[:, 1] * self.lon_range + self.lon_min
                plt.plot(v_lon, v_lat, color=c2, linewidth=0.8)
        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()

    def _compare_sequence(self, standard, test,threshold):
        standard_mean,standard_std = standard
        test_mean,test_std = test
        assert standard_mean.size() == test_mean.size()
        assert standard_std.size() == test_std.size()
        length = standard_mean.size()[0]
        # print(torch.stack([model_seq, test_seq], dim=0).size())
        result = np.zeros(length)
        for idx in range(length):
            if abs(standard_mean[idx]-test_mean[idx])>0.05 and abs(standard_std[idx]-test_std[idx]) > 0.05:
                result[idx] = 1
        return sum(result) / length > threshold


