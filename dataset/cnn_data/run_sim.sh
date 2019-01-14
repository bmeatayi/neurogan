#!/bin/bash
# SBATCH -o /home/hpc/pn56ju/ga48zop2/log_jobs/myjob.%j.%N.out
# SBATCH -D /home/hpc/pn56ju/ga48zop2/neurogan/examples/
# SBATCH -J cnn_simulate
# SBATCH --clusters=serial
# SBATCH --get-user-env
# SBATCH --mail-type=end
# SBATCH --mem=12000mb
# SBATCH --mail-user=m.atayi@tum.de
# SBATCH --export=NONE
# SBATCH --time=07:00:00
source /etc/profile.d/modules.sh
module load python
source activate rvdev
cd /home/hpc/pn56ju/ga48zop2/neurogan/dataset/cnn_data
python cnn_simulate.py
