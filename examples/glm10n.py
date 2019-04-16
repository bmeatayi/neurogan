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


spike_file = '..//dataset//GLM_2D_10n_shared_noise//data.npy'
stim_file = '..//dataset//GLM_2D_10n_shared_noise//stim.npy'

log_folder = 'glm10n_shared_noise//run15_rebar_lr1e-4_latentdim3_9layers//'
is_shared_noise = True
batch_size = 128
N = 10
nt = 20
nx = 30

train_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                           stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(0, 14000))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                         stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(14000, 14999))

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

generator = GeneratorGLM(latent_dim=3, stim_shape=(nt, nx),
                         out_shape=(1, N),
                         sampler=torch.distributions.bernoulli.Bernoulli,
                         is_shared_noise=is_shared_noise)

discriminator = Discriminator(sp_dim=(1, N), stim_dim=(nt, nx),
                              n_units=[128, 256, 256, 256, 256, 256, 256, 128, 64],
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
             lr=1e-4, b1=.5, b2=0.999,
             log_interval=10000, n_epochs=8000,
             n_disc_train=5,
             temp_anneal=1.0
             )

print(generator.shn_layer.weight)
torch.save(discriminator, log_folder + 'discriminator.pt')
