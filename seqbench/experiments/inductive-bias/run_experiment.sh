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

find_project_root() {
  local current_dir="$PWD"
  while [[ "$current_dir" != "/" ]]; do
    if [[ -f "$current_dir/.seqbench_marker" ]]; then
      echo "$current_dir"
      return 0
    fi
    current_dir="$(dirname "$current_dir")"
  done
  echo "Error: Could not find project root" >&2
  return 1
}


PROJECT_ROOT=$(find_project_root) || exit 1
EXPERIMENTS_DIR="experiments"
SCRIPT_DIR="${0%/*}"

# Extract the experiment folder name
EXPERIMENT_NAME=$(basename "$SCRIPT_DIR")

python3 "$PROJECT_ROOT/masterscript.py" --generator-hyperparameters $PROJECT_ROOT/"$EXPERIMENTS_DIR"/"$EXPERIMENT_NAME"/experiment-config/data_config.yaml \
    --window-slider-kwargs "$PROJECT_ROOT/config/window_slider.yaml"\
    --regressor pyseq.models.regressors.linear_regression.LinearRegressionModel\
    --thresholder-kwargs "$EXPERIMENTS_DIR"/"$EXPERIMENT_NAME"/experiment-config/wald-constant-thresholder.yaml\
    --scorer-kwargs "$EXPERIMENTS_DIR"/"$EXPERIMENT_NAME"/experiment-config/cusum.yaml\
    --experiment-name "$EXPERIMENT_NAME"\
    --write-results

python3 "$PROJECT_ROOT/masterscript.py" --generator-hyperparameters $PROJECT_ROOT/"$EXPERIMENTS_DIR"/"$EXPERIMENT_NAME"/experiment-config/data_config.yaml \
    --window-slider-kwargs "$PROJECT_ROOT/config/window_slider.yaml"\
    --regressor pyseq.models.regressors.gaussian_process.GPRModel\
    --thresholder-kwargs "$EXPERIMENTS_DIR"/"$EXPERIMENT_NAME"/experiment-config/wald-constant-thresholder.yaml\
    --scorer-kwargs "$EXPERIMENTS_DIR"/"$EXPERIMENT_NAME"/experiment-config/cusum.yaml\
    --experiment-name "$EXPERIMENT_NAME"\
    --write-results

