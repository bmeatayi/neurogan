#!/bin/bash

#SBATCH -o /home/hpc/pn56ju/ga48zop2/log_jobs/myjob.%j.%N.out

#SBATCH -D /home/hpc/pn56ju/ga48zop2/neurogan/examples/

#SBATCH -J discrete-cgan-training

#SBATCH --clusters=serial

#SBATCH --get-user-env

#SBATCH --mail-type=end
#SBATCH --mem=8000mb

#SBATCH --mail-user=m.atayi@tum.de

#SBATCH --export=NONE

#SBATCH --time=05:00:00

source /etc/profile.d/modules.sh

module load python

source activate rvdev

cd /home/hpc/pn56ju/ga48zop2/neurogan/examples/

python glm30n.py

# wgan-gp with gs (default)
#python run_cgan.py --n_epochs 1000 --gs_temp 0.5 --log_dir cgan_results//SharedNoise_30N_run02_lam.1_gs_temp1// --spike_file ..//dataset//GLM_2D_30n_shared_noise//data.npy --stim_file ..//dataset//GLM_2D_30n_shared_noise//stim.npy

# wgan-gp with REINFORCE
# python run_cgan.py --lr 0.0001 --n_epochs 2000 --grad_mode reinforce --log_dir cgan_results//reinforce__30N_run01// --spike_file ..//dataset//GLM_2D_30n_shared_noise//data.npy --stim_file ..//dataset//GLM_2D_30n_shared_noise//stim.npy

# JS with REINFORCE
#python run_cgan.py --lr 0.0001 --n_epochs 2000 --grad_mode reinforce --log_dir cgan_results//reinforce__30N_run01// --spike_file ..//dataset//GLM_2D_30n_shared_noise//data.npy --stim_file ..//dataset//GLM_2D_30n_shared_noise//stim.npy
