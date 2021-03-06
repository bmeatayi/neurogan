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

log_folder = 'glm30n//run02_rebar_lr5e-4_hplr1e-3_deeperDiscr//'

batch_size = 128
N = 30
nt = 30
nx = 40

train_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                           stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(0, 13000))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                         stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(13000, 13999))

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

generator = GeneratorGLM(latent_dim=1, stim_shape=(nt, nx),
                         out_shape=(1, N),
                         sampler=torch.distributions.bernoulli.Bernoulli)

discriminator = Discriminator(sp_dim=(1, N), stim_dim=(nt, nx),
                              n_units=[128, 256, 512, 512, 512, 256, 128],
                              mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                              p_drop=None,
                              spectral_norm=True)

solver = TrainerCGAN(optimizer_g=torch.optim.Adam,
                     optimizer_d=torch.optim.Adam,
                     log_folder=log_folder,
                     gan_mode='sn',
                     lambda_gp=None,
                     grad_mode='rebar',
                     gs_temp=1,
                     n_neuron=N)

solver.train(generator=generator, discriminator=discriminator,
             train_loader=train_dataloader, val_loader=val_dataloader,
             lr=5e-4, b1=.5, b2=0.999,
             log_interval=5000, n_epochs=2500,
             n_disc_train=5,
             temp_anneal=1.0
             )

print(generator.shn_layer.weight)
torch.save(discriminator, log_folder + 'discriminator.pt')
