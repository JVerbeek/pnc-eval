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

python3 ../../masterscript.py --generator-hyperparameters experiment-config/data_config.yaml \
    --window-slider-kwargs ../../config/window_slider.yaml\
    --regressor pyseq.models.regressors.linear_regression.LinearRegressionModel\
    --thresholder-kwargs experiment-config/wald-constant-thresholder.yaml\
    --scorer-kwargs experiment-config/cusum.yaml\
    --experiment-name "inductive-bias"\
    --write-results

python3 ../../masterscript.py --generator-hyperparameters experiment-config/data_config.yaml \
    --window-slider-kwargs ../../config/window_slider.yaml\
    --regressor pyseq.models.regressors.gaussian_process.GPRModel\
    --thresholder-kwargs experiment-config/wald-constant-thresholder.yaml\
    --scorer-kwargs experiment-config/cusum.yaml\
    --experiment-name "inductive-bias"\
    --write-results
