#!/bin/bash

#SBATCH --account=nn9851k
#SBATCH --job-name=UD_EVAL
#SBATCH --partition=a100
#SBATCH --gpus-per-node=1
#SBATCH --time=2:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=report/%j.out

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

####################################################
### Assumes that the datasets are already downloaded
# Otherwise run download.sh
####################################################

# Load modules
module --quiet purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/

# Include this if running on A100
module --force swap StdEnv Zen2Env 

module load nlpl-nlptools/02-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-wandb/0.15.2-foss-2022b-Python-3.10.8 		
module load nlpl-transformers/4.38.2-foss-2022b-Python-3.10.8
module load nlpl-cython/0.29.37-foss-2022b-Python-3.10.8    # for dependency_decoding

# Activate virtual environment
#source /cluster/work/users/vlhandfo/HPLT-WP4/venv/bin/activate

python3 train.py "$@"
