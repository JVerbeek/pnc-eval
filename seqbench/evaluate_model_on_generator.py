
import argparse
import importlib
import os
import yaml
import sys
import numpy as np

from pyseq.models.base_stack_detector import StackDetector
from run_dataset_generation import make_dataset


def import_object_from_string(dotted_path):
    """Import an object (function, class, etc.) from a dotted module path like 'module.submodule.func'."""
    module_path, obj_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def handle_open_file(filename):
    try:
        if filename is None:
            return {}
        with open(filename, "r") as f:
                kwargs = yaml.safe_load(f)
        return kwargs
    except FileNotFoundError:
        print(f"File not found for {filename}. Did you supply the correct path?")
        sys.exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator-hyperparameters', default=None)
    parser.add_argument('--regressor', required=True, help='Dotted path to the regressor class')
    parser.add_argument('--regressor-hyperparameters', default=None, help='YAML file of keyword arguments for the regressor class (default: None, use empty dict)')
    parser.add_argument('--window-slider', default='pyseq.models.window_sliders.window_slide.WindowSlider', help='Dotted path to the window slider class (default: WindowSlider)')
    parser.add_argument('--window-slider-kwargs', default=None, help='YAML file of keyword arguments for the window slider class (default: None, use empty dict)')
    parser.add_argument('--thresholder', default='pyseq.models.thresholders.wald_constant_thresholder.WaldConstantThresholder', help='Dotted path to the thresholder class (default: WaldConstantThresholder)')
    parser.add_argument('--thresholder-kwargs', default=None, help='YAML file of keyword arguments for the thresholder class (default: None, use empty dict)')
    parser.add_argument('--scorer', default='pyseq.models.scorers.cusum.BidirectionalCUSUMScorer', help='Score function to convert regression output to scores (default: cusum)')
    parser.add_argument('--plot-test-results', action='store_true', help='Whether to plot test results (default: False)')

    args = parser.parse_args()

    # Use the generator and model
    # Parse model and generator kwargs from YAML 
    # If the files are not provided, use empty dicts as kwargs
    # Load regressor, window-slider, and thresholder hyperparameters
    # TODO: add error handling for YAML loading
    regressor_kwargs = handle_open_file(args.regressor_hyperparameters)
    window_slider_kwargs = handle_open_file(args.window_slider_kwargs)
    thresholder_kwargs = handle_open_file(args.thresholder_kwargs)
    
    # Import window slider, regressor, thresholder, scorer
    regressor_cls = import_object_from_string(args.regressor)
    regressor = regressor_cls(**regressor_kwargs)
    #metric = import_object_from_string(args.metric)
    scorer = import_object_from_string(args.scorer)()

    window_slider_cls = import_object_from_string(args.window_slider)
    window_slider = window_slider_cls(window_size=1, skip_length=1)
    thresholder_cls = import_object_from_string(args.thresholder)
    thresholder = thresholder_cls(**thresholder_kwargs)

    sd = StackDetector(window_slider=window_slider, regressor=regressor, 
                       thresholder=thresholder, 
                       scorer=scorer)

    # Training (only if model is fittable)
    if regressor.fittable:
        print("Model is fittable, training...")
        # If we want to add preprocessing steps, add them to this function call
        X_train, y_train, cps = make_dataset(generator_hyperparameters=args.generator_hyperparameters, generator_name="test", set_name="train")
        sd.fit(X_train, y_train)


    # Validation for hyperparameter selection (not implemented yet)
    #TODO: Implement hyperparameter selection
    # Testing
    X_test, y_test, cps = make_dataset(generator_hyperparameters=args.generator_hyperparameters, generator_name="test", set_name="test")

    pred_test = sd.predict(X_test, y_test)

    # optional test plotting:

    if args.plot_test_results:
        import matplotlib.pyplot as plt
        for t, y_t, cp, scores, regressor_preds in zip(t_test_s, y_test_s, test_pred_s, scores_s, regressor_pred_s):
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))
            ax[0].plot(t, y_t, "kx", markersize=20, label="data")
            # ax.plot(X_t[:len(cp)], cp, linestyle="-", linewidth=3, color="r", label="changepoint locs")
            ax[0].plot(t, regressor_preds, linestyle="-", linewidth=3, color="r", label="regressor prediction")
            ax[1].plot(t, scores, linestyle=":", linewidth=3, color="b", label="cusum score")
            ax[1].axhline(-np.log(thresholder.alpha), linestyle="-", linewidth=3, color="k", label="wald constant threshold")
            ax[0].set_xlabel("t", fontsize=30)
            ax[0].tick_params(axis='both', which='major', labelsize=15)
            ax[1].tick_params(axis='both', which='major', labelsize=15)
            ax[0].set_ylabel("y", fontsize=30)
            ax[1].set_xlabel("t", fontsize=30)
            ax[1].set_ylabel("score", fontsize=30)
            ax[0].fill_between(t.flatten(), ax[0].get_ylim()[0], ax[0].get_ylim()[1], where=scores > -np.log(thresholder.alpha), color="red", alpha=0.3, label="CUSUM > threshold")
            ax[1].fill_between(t.flatten(), ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=scores > -np.log(thresholder.alpha), color="red", alpha=0.3, label="CUSUM > threshold")
            #plt.suptitle("GP regression on gradual frequency change in oscillating data", fontsize=30)
            plt.legend(fontsize=15)
            plt.tight_layout()
            plt.show()


# For testing purposes, provide defaults if not running as a script
if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--generator-hyperparameters", "/home/janneke/repos/pnc-eval/pyseq_data/src/pyseq_data/example_config.yaml",
        #"--regressor", "src.models.regressors.linear_regression.LinearRegressionModel",
        "--regressor", "pyseq.models.regressors.random_forest_regression.MultiOutputRandomForest",
        #"--regressor-hyperparameters", "path/to/model_hyperparams.yaml",
        # "--thresholder-kwargs", "config/wald-constant-thresholder.yaml",
        "--plot-test-results"
    ]
    main()
