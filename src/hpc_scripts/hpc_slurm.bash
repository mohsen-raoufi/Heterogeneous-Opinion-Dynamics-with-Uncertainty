#!/bin/bash
#SBATCH --job-name=scioi_colab_27_35_job_starter
#SBATCH --output=job_starter_%j.txt # output file
#SBATCH --error=job_starter_%j.err  # error file
#SBATCH --partition=ex_scioi_node # partition to submit to
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-50:00 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000 # memory in MB per cpu allocated

source $HOME/miniconda3/bin/activate
conda activate domip

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/

# ## Change this address according to your own path
python $HOME/colab/collective-decison-making-with-direl/src/experiment_runner.py

exit
