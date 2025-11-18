#!/bin/sh
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=PGR-Standard

cd  /home/s2893001/projects/pconv
conda activate ./.conda
python src/lightning_bench_mnist.py
