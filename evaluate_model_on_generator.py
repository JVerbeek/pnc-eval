
import argparse
import importlib
import os
import yaml
import sys
import numpy as np


from src.models.base_stack_detector import StackDetector

from src.data_generators.generate_dataset import generate_dataset

def import_object_from_string(dotted_path):
    """Import an object (function, class, etc.) from a dotted module path like 'module.submodule.func'."""
    module_path, obj_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', default="src.data_generators.generation_script.generate_datasets", required=True, help='Dotted path to the generator function')
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
    # TODO: add error handling for YAML loading
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
                       scorer=scorer, prediction_window_size=10)

    # Training (only if model is fittable)
    if regressor.fittable:
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
        for X_t, y_t, rp, scores in zip(X_test, y_test, pred_test, scores):
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))
            cp_plot = np.concatenate((scores, np.zeros(((sd.window_slider.window_size - sd.window_slider.skip_length),))))
            ax[0].plot(X_t, y_t, "kx", markersize=20, label="data")
            # ax.plot(X_t[:len(cp)], cp, linestyle="-", linewidth=3, color="r", label="changepoint locs")
            ax[0].plot(X_t, np.concatenate((rp, np.zeros(len(X_t)-len(cp)))), linestyle="-", linewidth=3, color="r", label="regressor prediction")
            ax[1].plot(X_t, np.concatenate((scores, np.zeros((sd.window_slider.window_size-sd.window_slider.skip_length),))), linestyle=":", linewidth=3, color="b", label="cusum score")
            ax[1].axhline(-np.log(thresholder.alpha), linestyle="-", linewidth=3, color="k", label="wald constant threshold")
            ax[0].set_xlabel("t", fontsize=30)
            ax[0].tick_params(axis='both', which='major', labelsize=15)
            ax[1].tick_params(axis='both', which='major', labelsize=15)
            ax[0].set_ylabel("y", fontsize=30)
            ax[1].set_xlabel("t", fontsize=30)
            ax[1].set_ylabel("score", fontsize=30)
            ax[0].fill_between(X_t.flatten(), ax[0].get_ylim()[0], ax[0].get_ylim()[1], where=cp_plot > 0, color="red", alpha=0.3, label="CUSUM > threshold")
            ax[1].fill_between(X_t.flatten(), ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=cp_plot > 0, color="red", alpha=0.3, label="CUSUM > threshold")
            plt.suptitle("GP regression on gradual frequency change in oscillating data", fontsize=30)
            plt.legend(fontsize=15)
            plt.savefig("/home/janneke/repos/pnc-eval/constant-change-cusum.png", dpi=300)
            plt.show()


# For testing purposes, provide defaults if not running as a script
if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--generator", "src.data_generators.generation_script.generate_datasets",
        "--generator-hyperparameters", "config/amplitude-stepped-cons.yaml",
        #"--regressor", "src.models.regressors.linear_regression.LinearRegressionModel",
        "--regressor", "src.models.regressors.random_forest_regression.MultiOutputRandomForest",
        #"--regressor-hyperparameters", "path/to/model_hyperparams.yaml",
        "--window-slider-kwargs", "config/window_slider.yaml",
        "--thresholder-kwargs", "config/wald-constant-thresholder.yaml",
        "--plot-test-results"
    ]
    main()
