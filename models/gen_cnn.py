import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GeneratorCNN(nn.Module):
    def __init__(self, nw=40, nh=40, nl=5,
                 n_filters=(15, 9),
                 kernel_size=(15, 11),
                 n_cell=1,
                 sampler=torch.distributions.bernoulli.Bernoulli):
        super(GeneratorCNN, self).__init__()
        self.nW = nw
        self.nH = nh
        self.nL = nl
        self.nFiltL1 = n_filters[0]
        self.nFiltL2 = n_filters[1]
        self.szFiltL1 = kernel_size[0]
        self.szFiltL2 = kernel_size[1]
        self.n_cell = n_cell
        self.sampler = sampler
        self.latent_dim = 3
        self.n_t = 1  # Number of spikes bin to be predicted by generator

        self.l3_filt_shape = None

        self.conv1 = nn.Conv2d(in_channels=self.nL,
                               out_channels=self.nFiltL1,
                               kernel_size=(self.szFiltL1, self.szFiltL1),
                               stride=1,
                               padding=0,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=self.nFiltL1,
                               out_channels=self.nFiltL2,
                               kernel_size=(self.szFiltL2, self.szFiltL2),
                               stride=1,
                               padding=0,
                               bias=True)

        in_shp, conv2outShape = self._compute_fc_in()
        self.l3_filt_shape = (self.n_cell, *conv2outShape)

        self.fc = nn.Linear(in_features=in_shp,
                            out_features=self.n_cell,
                            bias=True)
        self.shared_noise = Parameter(torch.ones(3).fill_(.05))

    def _compute_fc_in(self):
        x = np.random.random([1, self.nL, self.nW, self.nH])
        x = torch.from_numpy(x)
        x = self.conv1(x.float())
        x = self.conv2(x)
        conv2shape = x.size()[1:]
        x = x.view(x.size(0), -1)
        return x.size(1), conv2shape

    def forward(self, z, stim):
        x_conv1 = self.conv1(stim)
        x = F.relu(x_conv1 + self.shared_noise[0] * z[:,0].view(-1,1,1,1).repeat((1, *x_conv1.shape[1:])))
        x_conv2 = self.conv2(x)
        x = F.relu(x_conv2 + self.shared_noise[1] * z[:,1].view(-1,1,1,1).repeat((1, *x_conv2.shape[1:])))
        x = self.fc(x.view([x.shape[0], -1])) + (self.shared_noise[2] * z[:,2]).unsqueeze(1)
        return x.unsqueeze(1)

    def generate(self, z, stim):
        r"""
                Generate spikes
                Args:
                    z (tensor): input noise
                    stim (tensor): stimulus

                Returns:
                    spikes (tensor): discrete spikes
                """
        with torch.no_grad():
            logits = self.forward(z, stim)
            sampling_distribution = self.sampler(logits=logits)
            spikes = sampling_distribution.sample()
        return spikes