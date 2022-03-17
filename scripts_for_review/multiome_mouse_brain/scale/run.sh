#!/bin/bash
  
#SBATCH -p gpu,gpu24,gpu96
#SBATCH --gres gpu:1
#SBATCH --mem 100G
#SBATCH -t 48:00:00
#SBATCH -J scale
#SBATCH -o scale.%j.out

cd /home/yuanh/analysis/sc_basset/10x_ARC_mouse_brain/competitors/scale
source /home/yuanh/.bashrc
source activate scvi

SCALE.py -d data/ -t -x 0.06 -o output --min_peaks 600 --impute
