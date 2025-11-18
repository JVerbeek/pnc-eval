#!/bin/bash
#SBATCH --partition=das
#SBATCH --qos=das-small
#SBATCH --cpus-per-task= 4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time= 10:00:00
#SBATCH --output=%A_%a.out
#SBATCH --error=%A-%a.err
#SBATCH --array=0-1


config=experiments/noise-stepped-osc/job-array-config.txt

model=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
generator=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

evaluate_model_on_generator.py --generator-hyperparameters generator --model-hyperparameters model