#!/bin/bash
#SBATCH --job-name=PY_PostPro
#SBATCH --output=PostPro_%j.txt # output file
#SBATCH --error=PostPro_%j.err  # error file
#SBATCH --partition=ex_scioi_node # partition to submit to
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=5-10:00 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=20000 # memory in MB per cpu allocated

source $HOME/miniconda3/bin/activate
conda activate domip

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/

# python $HOME/colab/collective-decison-making-with-direl/src/networkStudy_parallel_eigValTime.py
python $HOME/colab/collective-decison-making-with-direl/src/networkStudy_parallel.py

exit
