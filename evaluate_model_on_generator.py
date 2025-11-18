
import argparse
import importlib
import json
import os
import yaml
import sys
import numpy as np

from src.models.base_stack_detector import StackDetector


def import_object_from_string(dotted_path):
    """Import an object (function, class, etc.) from a dotted module path like 'module.submodule.func'."""
    module_path, obj_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True, help='Dotted path to the generator function')
    parser.add_argument('--generator-hyperparameters', default=None, help='YAML file of keyword arguments for the generator function (default: None, use empty dict)') 
    parser.add_argument('--regressor', required=True, help='Dotted path to the regressor class')
    parser.add_argument('--regressor-hyperparameters', default=None, help='YAML file of keyword arguments for the regressor class (default: None, use empty dict)')
    parser.add_argument('--window-slider', default='src.models.window_sliders.window_slide.WindowSlider', help='Dotted path to the window slider class (default: WindowSlider)')
    parser.add_argument('--window-slider-kwargs', default=None, help='YAML file of keyword arguments for the window slider class (default: None, use empty dict)')
    parser.add_argument('--thresholder', default='src.models.thresholders.wald_constant_thresholder.WaldConstantThresholder', help='Dotted path to the thresholder class (default: WaldConstantThresholder)')
    parser.add_argument('--thresholder-kwargs', default=None, help='YAML file of keyword arguments for the thresholder class (default: None, use empty dict)')
    parser.add_argument('--scorer', default='src.models.scorers.cusum.BidirectionalCUSUMScorer', help='Score function to convert regression output to scores (default: cusum)')
    parser.add_argument('--plot-test-results', action='store_true', help='Whether to plot test results (default: False)')

    args = parser.parse_args()

    # Use the generator and model
    # Parse model and generator kwargs from YAML 
    # If the files are not provided, use empty dicts as kwargs
    # Load regressor, window-slider, and thresholder hyperparameters
    if args.regressor_hyperparameters and os.path.isfile(args.regressor_hyperparameters):
        with open(args.regressor_hyperparameters, "r") as f:
            regressor_kwargs = yaml.safe_load(f) or {}
    else:
        regressor_kwargs = {}

    if args.window_slider_kwargs and os.path.isfile(args.window_slider_kwargs):
        with open(args.window_slider_kwargs, "r") as f:
            window_slider_kwargs = yaml.safe_load(f) or {}
    else:
        window_slider_kwargs = {}

    if args.thresholder_kwargs and os.path.isfile(args.thresholder_kwargs):
        with open(args.thresholder_kwargs, "r") as f:
            thresholder_kwargs = yaml.safe_load(f) or {}
    else:
        thresholder_kwargs = {}

    # Load generator hyperparameters
    if args.generator_hyperparameters and os.path.isfile(args.generator_hyperparameters):
        with open(args.generator_hyperparameters, "r") as f:
            generator_kwargs = yaml.safe_load(f) or {}
    else:
        generator_kwargs = {}

    # Import data generator
    generator_fn = import_object_from_string(args.generator)


    # Import window slider, regressor, thresholder, scorer
    regressor_cls = import_object_from_string(args.regressor)
    regressor = regressor_cls(**regressor_kwargs)
    #metric = import_object_from_string(args.metric)
    scorer = import_object_from_string(args.scorer)()

    window_slider_cls = import_object_from_string(args.window_slider)
    window_slider = window_slider_cls(**window_slider_kwargs)
    thresholder_cls = import_object_from_string(args.thresholder)
    thresholder = thresholder_cls(**thresholder_kwargs)

    sd = StackDetector(window_slider=window_slider, regressor=regressor, 
                       thresholder=thresholder, 
                       scorer=scorer, prediction_window_size=10, verbose=False)

    # Training (only if model is fittable)
    if regressor.is_fittable:
        print("Model is fittable, training...")
        # If we want to add preprocessing steps, add them to this function call
        X_train, y_train, cps = generate_dataset(generator_kwargs, generator_fn, generator_hyperparameters=args.generator_hyperparameters, generator_name=args.generator, set_name="train")

        sd.fit(X_train, y_train)


    # Validation for hyperparameter selection (not implemented yet)
    #TODO: Implement hyperparameter selection

    # Testing

    X_test, y_test, cps = generate_dataset(generator_kwargs, generator_fn, generator_hyperparameters=args.generator_hyperparameters, generator_name=args.generator, set_name="test")


    pred_test = sd.predict(X_test, y_test)

    # optional test plotting:
    if args.plot_test_results:
        import matplotlib.pyplot as plt 

        for X_t, y_t, p in zip(X_test, y_test, pred_test):
            fig, ax = plt.subplots(1, 1)
            ax.plot(X_t, y_t)
            ax.plot(X_t[:len(p)], p)
            ax.fill_between(X_t.flatten()[sd.window_slider.window_size - sd.window_slider.skip_length:], ax.get_ylim()[0], ax.get_ylim()[1], where=p > 0, color="red", alpha=0.3)
            plt.show()

def generate_dataset(generator_kwargs, generator_fn, generator_hyperparameters, generator_name, set_name="train"):
    # Check if data has been generated before:

    # Convert hyperparameter kwargs dict to a folder name
    if generator_hyperparameters:
        # Create a concise, readable string from generator_kwargs for folder naming
        import hashlib
        # Use a hash of the kwargs string for a robust folder 
        generator_kwargs_str = json.dumps(generator_kwargs, sort_keys=True)
        
        hash_object = hashlib.sha256(generator_kwargs_str.encode())
        generator_kwargs_str = hash_object.hexdigest()[:32] #technically not unique, but hopefully fine

    else: # no hyperparameters provided, e.g. args.generator_hyperparameters is None
        generator_kwargs_str = "default"

    generated_data_folder = os.path.join("generated_data", generator_name.replace('.', '_'), generator_kwargs_str)

    os.makedirs(generated_data_folder, exist_ok=True)

    X_file = os.path.join(generated_data_folder, f"X_{set_name}.npz")
    y_file = os.path.join(generated_data_folder, f"y_{set_name}.npz")
    cps_file = os.path.join(generated_data_folder, f"cps_{set_name}.npz")
    params_file = os.path.join(generated_data_folder, f"params_{set_name}.json")


    if not os.path.exists(X_file) or not os.path.exists(y_file) or not os.path.exists(cps_file) or not os.path.exists(params_file):
        X, y, cps, params = generator_fn(properties=generator_kwargs)        # note:X_train is a N_d long list of matrices, Y_train is a N_d long list of indices of singular change points
        # Save the generated data with explicit keys
        np.savez_compressed(X_file, X=X) #check if this works for lists
        np.savez_compressed(y_file, y=y)
        np.savez_compressed(cps_file, cps=cps)
        # Save params as JSON (assume params is a plain dict)
        #print(params)
        #with open(params_file, "w") as f:
        #    json.dump(params, f, indent=2)
    else:
    # Load the generated data using explicit keys
        X = np.load(X_file)["X"]
        y = np.load(y_file)["y"]
        cps = np.load(cps_file)["cps"]
        #with open(params_file, "r") as f:
        #    params = json.load(f)



    # Currently always standardize the y data, could implement generic preprocessing later?
    y = [(y_instance - y_instance.mean())/y_instance.std() for y_instance in y]

    return X, y, cps#, params

# For testing purposes, provide defaults if not running as a script
if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--generator", "src.data_generators.generation_script.generate_datasets",
        "--generator-hyperparameters", "config/amplitude-stepped-cons.yaml",
        "--regressor", "src.models.regressors.linear_regression.LinearRegressionModel",
        #"--regressor-hyperparameters", "path/to/model_hyperparams.yaml",
        "--window-slider-kwargs", "config/window_slider.yaml",
        "--thresholder-kwargs", "config/wald-constant-thresholder.yaml",
        "--plot-test-results"
    ]
    main()
