#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -cwd                  
#$ -N mnist_bench
#$ -q gpu 
#$ -l gpu=1
#$ -l h_rss=32G
#$ -l h_vmem=32G 
#$ -l h_rt=00:30:00 

exec > "logs/mnist_bench_${JOB_ID}.${SGE_TASK_ID}.out" 2> "logs/mnist_bench_${JOB_ID}.${SGE_TASK_ID}.err"
echo "Job hosted on: $(hostname)"
 
# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load cuda
echo "cuda loaded successfully, checking GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda
conda activate ./.conda
echo "Conda env successfully activated"

# Run the program
python cli.py bench-mnist-array --num_array $SGE_TASK_ID --config_json ./hp_grids/mnist.json
