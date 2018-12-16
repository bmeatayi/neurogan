import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from models.gen_glm import GeneratorGLM
from models.disc_mlp import Discriminator
from utils.trainer import TrainerCGAN
from utils.cgan_dataset import GanDataset

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
plt.ioff()

spike_file = '..//dataset//GLM_2D_30n_shared_noise//data.npy'
stim_file = '..//dataset//GLM_2D_30n_shared_noise//stim.npy'

log_folder = 'cgan_results//SharedNoise_30N_run03_lam.1_temp.5//'

batch_size = 128
N = 30

train_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                           stim_win_len=30, cnt_win_len=0, n_split=1, st_end=(0, 13000))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                         stim_win_len=30, cnt_win_len=0, n_split=1, st_end=(13000, 13999))

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

generator = GeneratorGLM(latent_dim=1, stim_shape=(30, 40),
                         out_shape=(1, N),
                         sampler=torch.distributions.bernoulli.Bernoulli)

discriminator = Discriminator(sp_dim=(1, N), stim_dim=(30, 40),
                              n_units=[128, 256, 512, 256, 128],
                              mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                              p_drop=None)

solver = TrainerCGAN(optimizer_g=torch.optim.Adam,
                     optimizer_d=torch.optim.Adam,
                     log_folder=log_folder,
                     gan_mode='wgan-gp',
                     lambda_gp=.1,
                     grad_mode='gs',
                     gs_temp=.5,
                     n_neuron=N)

solver.train(generator=generator, discriminator=discriminator,
             train_loader=train_dataloader, val_loader=val_dataloader,
             lr=0.0003, b1=.5, b2=0.999,
             log_interval=10000, n_epochs=2000,
             n_disc_train=5
             )

print(generator.shn_layer.weight)
