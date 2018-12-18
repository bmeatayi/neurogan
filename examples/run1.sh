#!/bin/bash

#SBATCH -o /home/hpc/pn56ju/ga48zop2/log_jobs/myjob.%j.%N.out

#SBATCH -D /home/hpc/pn56ju/ga48zop2/neurogan/examples/

#SBATCH -J discrete-cgan-training

#SBATCH --clusters=serial

#SBATCH --get-user-env

#SBATCH --mail-type=end
#SBATCH --mem=6000mb

#SBATCH --mail-user=m.atayi@tum.de

#SBATCH --export=NONE

#SBATCH --time=08:00:00

source /etc/profile.d/modules.sh

module load python

source activate rvdev

cd /home/hpc/pn56ju/ga48zop2/neurogan/examples/

#python glm30n.py

python run_cgan.py --spike_file ..//dataset//GLM_2D_30n_shared_noise//data.npy --stim_file ..//dataset//GLM_2D_30n_shared_noise//stim.npy --log_dir cgan_results//SharedNoise_30N_run07_lam.1_gs//


