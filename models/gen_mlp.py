"""
Generators with multilayer perceptron architecture for conditional GANs

Author: Mohamad Atayi
"""

import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=1,
                 stim_shape=(1, 1, 1),
                 sp_shape=(1, 1),
                 n_units=[128, 256, 128],
                 mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                 bn_momentum=0.8,
                 sampler=torch.distributions.bernoulli.Bernoulli):
        r"""
        Generator with multilayer perceptron architecture
        Args:
            latent_dim (int): latent noise dimension
            stim_shape (tuple): shape of the stimulus
            sp_shape (tuple): shape of output spikes [#time_bin, #neurons]
            n_units (list): list of number of units in each layer
            mid_act_func (torch.nn.functional): activation function in the middle layers
            bn_momentum (float): batch normalization momentum (None for no batch normalization)
            sampler (torch.distributions): sampler function (either Bernoulli or binary Gumbel-Softmax)
        """
        super(Generator, self).__init__()

        self.in_dim = np.prod(stim_shape)
        self.latent_dim = np.prod(latent_dim)
        self.n_t, self.n_cell = sp_shape
        self.fcout_dim = np.prod(sp_shape)
        self.sampler = sampler

        dense_layers = OrderedDict()
        dense_layers['fcin'] = nn.Linear(latent_dim + self.in_dim, n_units[0])
        dense_layers['act0'] = mid_act_func
        for i in range(1, len(n_units)):
            print(i, n_units[i])
            dense_layers['fc'+str(i)] = nn.Linear(n_units[i-1], n_units[i])
            if bn_momentum is not None:
                dense_layers['bn'+str(i)] = nn.BatchNorm1d(n_units[i], momentum=bn_momentum)
            dense_layers['act'+str(i)] = mid_act_func
        dense_layers['fcout'] = nn.Linear(n_units[-1], self.fcout_dim)

        self.all_layers = nn.Sequential(dense_layers)
        print(self)

    def forward(self, z, stim):
        x = torch.cat((z, stim.view(stim.size(0), -1)), -1)
        x = self.all_layers(x)
        return x.view(x.size(0), self.n_t, self.n_cell)

    def generate(self, z, stim):
        r"""
        Generate spikes
        Args:
            z: input noise
            stim: stimulus

        Returns:
            discrete spikes
        """
        with torch.no_grad():
            logits = self.forward(z, stim)
            sampling_distribution = self.sampler(logits=logits)
        spikes = sampling_distribution.sample()
        return spikes

