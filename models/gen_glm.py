import torch.nn as nn
import numpy as np
import torch


class GeneratorGLM(nn.Module):
    def __init__(self, latent_dim=1, stim_shape=(1, 1, 1), out_shape=(1, 1),
                 sampler=torch.distributions.bernoulli.Bernoulli, is_shared_noise=False):
        r"""
        GLM based generator with shared noise
        Args:
            latent_dim (int): dimension of latent noise
            stim_shape (tuple): shape of stimuli
            out_shape (tuple): shape of output spikes
            sampler (torch.distributions): sampler function (either Bernoulli or binary Gumbel-Softmax)
        """
        super(GeneratorGLM, self).__init__()

        self.in_dim = np.prod(stim_shape)
        self.latent_dim = np.prod(latent_dim)
        self.n_t, self.n_cell = out_shape
        self.glm_out_shape = np.prod(out_shape)
        self.sampler = sampler
        self.is_shared_noise = is_shared_noise

        self.GLM = nn.Linear(self.in_dim, self.glm_out_shape)
        self.shn_layer = nn.Linear(self.latent_dim, self.n_cell, bias=False)
        print(self)

    def forward(self, z, stim):
        stim = stim.view(stim.size(0), -1)
        x = self.GLM(stim)
        if self.is_shared_noise:
            x = x + self.shn_layer(z)
        return x.view(x.size(0), self.n_t, self.n_cell)

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
