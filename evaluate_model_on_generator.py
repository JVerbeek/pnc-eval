
import argparse
import importlib
import json
import os
import yaml
import sys
import numpy as np

#Temporary import until parsing is added
from src.models.threshold_models.threshold_model import ThresholdModel


def import_object_from_string(dotted_path):
    """Import an object (function, class, etc.) from a dotted module path like 'module.submodule.func'."""
    module_path, obj_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True, help='Dotted path to the generator function')
    parser.add_argument('--generator-hyperparameters', default=None, help='YAML file of keyword arguments for the generator function (default: None, use empty dict)') 
    parser.add_argument('--model', required=True, help='Dotted path to the model function')
    parser.add_argument('--model-hyperparameters', default=None, help='YAML file of keyword arguments for the model class (default: None, use empty dict)')
    parser.add_argument('--metric', default='src.metrics.metrics.f1_score', help='Metric to optimize threshold on (default: f1)')
    parser.add_argument('--score-function', default='src.score_functions.cusum.cusum_score', help='Score function to convert regression output to scores (default: cusum)')

    args = parser.parse_args()

    # Use the generator and model
    # Parse model and generator kwargs from YAML 
    # If the files are not provided, use empty dicts as kwargs
    # Load model hyperparameters
    if args.model_hyperparameters and os.path.isfile(args.model_hyperparameters):
        with open(args.model_hyperparameters, "r") as f:
            model_kwargs = yaml.safe_load(f) or {}
    else:
        model_kwargs = {}

    # Load generator hyperparameters
    if args.generator_hyperparameters and os.path.isfile(args.generator_hyperparameters):
        with open(args.generator_hyperparameters, "r") as f:
            generator_kwargs = yaml.safe_load(f) or {}
    else:
        generator_kwargs = {}

    # Import generator function and model class
    generator_fn = import_object_from_string(args.generator)
    model_cls = import_object_from_string(args.model)
    metric = import_object_from_string(args.metric)
    score_function = import_object_from_string(args.score_function)

    # Training
    # Check if data has been generated before:

    # Convert hyperparameter kwargs dict to a folder name
    if args.generator_hyperparameters:
        # Create a concise, readable string from generator_kwargs for folder naming
        import hashlib
        # Use a hash of the kwargs string for a robust folder 
        generator_kwargs_str = json.dumps(generator_kwargs, sort_keys=True)
        
        hash_object = hashlib.sha256(generator_kwargs_str.encode())
        generator_kwargs_str = hash_object.hexdigest()[:32] #technically not unique, but hopefully fine

    else:
        generator_kwargs_str = "default"

    generated_data_folder = os.path.join("generated_data", args.generator.replace('.', '_'), generator_kwargs_str)

    X_file = os.path.join(generated_data_folder, "X_train.npz")
    y_file = os.path.join(generated_data_folder, "y_train.npz")
    os.makedirs(generated_data_folder, exist_ok=True)

    if not os.path.exists(X_file) or not os.path.exists(y_file):
        X_train, y_train = generator_fn(**generator_kwargs)
        # note:output of both X_train and y_train are n_D-by-t numpy arrays
        # Save the generated data with explicit keys
        np.savez_compressed(X_file, X_train=X_train)
        np.savez_compressed(y_file, y_train=y_train)
    else:
        # Load the generated data using explicit keys
        X_train = np.load(X_file)['X_train']
        y_train = np.load(y_file)['y_train']



    # Instantiate base model with kwargs
    base_model = model_cls(**model_kwargs)


    # TODO: add check if base_model needs to be fitted before passing to ThresholdModel
    # Check here

    # Wrap base model in ThresholdModel

    model = ThresholdModel(base_model, score_function, metric)    

    # Fit the threshold model
    model.fit(X_train, y_train)

    # Visualize the threshold optimization for testing purposes
    model.visualize_optimization()

# For testing purposes, provide defaults if not running as a script

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--generator", "src.data_generators.mean_change.generate_mean_change",
        "--model", "src.models.regression_models.linear_regression.LinearRegressionModel",
        # "--model-hyperparameters", "path/to/model_hyperparams.yaml",
        "--generator-hyperparameters", "mean_change_example.yaml"
    ]
    main()
