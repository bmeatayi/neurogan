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

# import matplotlib.pyplot as plt
# import matplotlib

spike_file = '..//dataset//GLM_1D_1n//data.npy'
stim_file = '..//dataset//GLM_1D_1n//stim.npy'

log_folder = 'glm_results//run12_rebarNEW_ndisc5_lr1e-3_hp_lr=1e-3//'

batch_size = 128
N = 1
nt = 20
nx = 1

train_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                           stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(0, 4000))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                         stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(4000, 4999))

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

generator = GeneratorGLM(latent_dim=1, stim_shape=(nt, nx),
                         out_shape=(1, N),
                         sampler=torch.distributions.bernoulli.Bernoulli)

discriminator = Discriminator(sp_dim=(1, N), stim_dim=(nt, nx),
                              n_units=[64, 128, 128, 64],
                              mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                              p_drop=None,
                              spectral_norm=True)

solver = TrainerCGAN(optimizer_g=torch.optim.Adam,
                     optimizer_d=torch.optim.Adam,
                     log_folder=log_folder,
                     gan_mode='sn',
                     lambda_gp=None,
                     grad_mode='rebar',
                     gs_temp=1.,
                     n_neuron=N)

solver.train(generator=generator, discriminator=discriminator,
             train_loader=train_dataloader, val_loader=val_dataloader,
             lr=1e-3, b1=.5, b2=0.999,
             log_interval=4000, n_epochs=4000,
             n_disc_train=5,
             temp_anneal=None
             )
