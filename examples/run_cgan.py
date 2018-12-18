import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.gen_glm import GeneratorGLM
from models.disc_mlp import Discriminator
from utils.trainer import TrainerCGAN
from utils.cgan_dataset import GanDataset

matplotlib.use('Agg')
plt.ioff()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike_file", type=str, default='..//dataset//GLM_2D_30n_shared_noise//data.npy', help="path of spike file")
    parser.add_argument("--stim_file", type=str, default='..//dataset//GLM_2D_30n_shared_noise//stim.npy', help="path of stimulation file")
    parser.add_argument("--log_dir", type=str, default='results', help="log directory")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--stim_win_len", type=int, default=30, help="stimulation window length")
    parser.add_argument("--cnt_win_len", type=int, default=0, help="spike count history")
    parser.add_argument("--n_split", type=int, default=1, help="number of splits (number spike bins in frame)")
    parser.add_argument("--st_end_train", type=tuple, default=(0, 13000), help="start and end frame of training set")
    parser.add_argument("--st_end_val", type=tuple, default=(13000, 13999), help="start and end frame of validation set")
    parser.add_argument("--latent_dim", type=int, default=1, help="latent dimension")
    parser.add_argument("--n_units_d", type=list, default=[128, 256, 512, 256, 128], help="Number of units in discriminator")
    parser.add_argument("--gan_mode", type=str, default='wgan-gp', help="Gan type:['wgan-gp', 'js', 'sn']")
    parser.add_argument("--grad_mode", type=str, default='gs', help="gradient estimator method: {'gs':Gumbel-Softmax,"
                                                                    ", 'reinforce': REINFORCE, 'rebar':REBAR")
    parser.add_argument("--lambda_gp", type=float, default=.1, help="gradient penalty scale in wgan-gp")
    parser.add_argument("--gs_temp", type=float, default=.5, help="Gumbel-Softmax temperature")
    parser.add_argument("--lr", type=float, default=.0003, help="learning rate")
    parser.add_argument("--log_interval", type=int, default=10000, help="logging interval")

    args = parser.parse_args()
    train_dataset = GanDataset(spike_file=args.spike_file,
                               stim_file=args.stim_file,
                               stim_win_len=args.stim_win_len,
                               cnt_win_len=args.cnt_win_len,
                               n_split=args.n_split,
                               st_end=args.st_end_train)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = GanDataset(spike_file=args.spike_file,
                             stim_file=args.stim_file,
                             stim_win_len=args.stim_win_len,
                             cnt_win_len=args.cnt_win_len,
                             n_split=args.n_split,
                             st_end=args.st_end_val)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    N = train_dataset.spike_count.shape[2]
    stim_shape = train_dataset[0][1].shape

    generator = GeneratorGLM(latent_dim=args.latent_dim,
                             stim_shape=stim_shape,
                             out_shape=(1, N),
                             sampler=torch.distributions.bernoulli.Bernoulli)

    discriminator = Discriminator(sp_dim=(1, N),
                                  stim_dim=stim_shape,
                                  n_units=args.n_units_d,
                                  mid_act_func=nn.LeakyReLU(0.2, inplace=True),
                                  p_drop=None)

    solver = TrainerCGAN(optimizer_g=torch.optim.Adam,
                         optimizer_d=torch.optim.Adam,
                         log_folder=args.log_dir,
                         gan_mode=args.gan_mode,
                         lambda_gp=args.lambda_gp,
                         grad_mode=args.grad_mode,
                         gs_temp=args.gs_temp,
                         n_neuron=N)

    solver.train(generator=generator,
                 discriminator=discriminator,
                 train_loader=train_dataloader,
                 val_loader=val_dataloader,
                 lr=args.lr,
                 b1=.5, b2=0.999,
                 log_interval=args.log_interval,
                 n_epochs=args.n_epochs,
                 n_disc_train=5
                 )
    print(generator.shn_layer.weight)
    torch.save(args.log_file + 'discriminator.pt', discriminator)
