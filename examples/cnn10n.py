import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from models.gen_cnn import GeneratorCNN
from models.disc_cnn import DiscriminatorCNN
from utils.trainer import TrainerCGAN
from utils.cgan_dataset import GanDataset

spike_file = '..//dataset//cnn_data//spike.npy'
stim_file = '..//dataset//cnn_data//stim.npy'

log_folder = 'cnn_results//run01_gs_anneal_//'

batch_size = 128
N = 10
nt = 5
nx = 40
f1sz = 11
f2sz = 7

train_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                           stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(0, 20000))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                         stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(24000, 24999))

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

generator = GeneratorCNN(nw=nx, nh=nx, nl=nt,
                         n_filters=(3, 2),
                         kernel_size=(f1sz, f2sz),
                         n_cell=N)

discriminator = DiscriminatorCNN(nw=nx, nh=nx, nl=nt,
                         n_filters=(3, 2),
                         kernel_size=(f1sz, f2sz),
                         n_cell=N)

solver = TrainerCGAN(optimizer_g=torch.optim.Adam,
                     optimizer_d=torch.optim.Adam,
                     log_folder=log_folder,
                     gan_mode='sn',
                     lambda_gp=None,
                     grad_mode='gs',
                     gs_temp=1,
                     n_neuron=N)

solver.train(generator=generator, discriminator=discriminator,
             train_loader=train_dataloader, val_loader=val_dataloader,
             lr=1e-4, b1=.5, b2=0.999,
             log_interval=10000, n_epochs=3500,
             n_disc_train=5,
             temp_anneal=.9995
             )

