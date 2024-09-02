#!/bin/sh
#SBATCH -J train_hscai
#SBATCH --time=1:00:00
#SBATCH --partition cas_v100_2
#SBATCH --comment pytorch
#SBATCH -N 1
#SBATCH --gres=gpu:2
module load python cuda/11.4 gcc/8.3.0
conda activate mlv-ir-od
python task/train.py \
    --epochs 10 \
    --batch 2 \
    --dataset "hscai"