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

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
torch.set_default_tensor_type(FloatTensor)

spike_file = '..//dataset//cnn_data//gabor//spike.npy'
stim_file = '..//dataset//cnn_data//gabor//stim.npy'

log_folder = 'cnn_results//run37_gabornew_gs1_ann.994_ndisc5_lr1e-4_shn[0,1]=0_6layers//'

batch_size = 128
N = 10
nt = 5
nx = 40
f1sz = 11
f2sz = 7

train_dataset = GanDataset(spike_file=spike_file, stim_file=stim_file,
                           stim_win_len=nt, cnt_win_len=0, n_split=1, st_end=(0, 24000))

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
                                 n_cell=N,
                                 spectral_norm=True,
                                 mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                                 n_units=[64, 128, 256, 256, 128, 64],
                                 p_drop=None
                                 )

solver = TrainerCGAN(optimizer_g=torch.optim.Adam,
                     optimizer_d=torch.optim.Adam,
                     log_folder=log_folder,
                     gan_mode='sn',
                     lambda_gp=None,
                     grad_mode='gs',
                     gs_temp=1,
                     n_neuron=N)

generator.shared_noise.data[0:2] = torch.tensor([0.,0.])
solver.train(generator=generator, discriminator=discriminator,
             train_loader=train_dataloader, val_loader=val_dataloader,
             lr=1e-4, b1=.5, b2=0.999,
             log_interval=2000, n_epochs=1000,
             n_disc_train=5,
             temp_anneal=.994
             )
