import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQBENCH_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(SEQBENCH_DIR)
import matplotlib.pyplot as plt
import numpy as np
from utils import handle_open_file, import_object_from_string, write_results
from plotting import plot_cusum_results

from pyseq.models.base_stack_detector import StackDetector
from run_dataset_generation import make_dataset

def generate_simple_mean_changes(path_to_data_config):
    t_test, y_test, cps = make_dataset(generator_hyperparameters=path_to_data_config, generator_name="simple_interpretable", set_name="simple_interpretable")
    return t_test, y_test, cps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator-hyperparameters', default=None)
    parser.add_argument('--regressor', required=True, help='Dotted path to the regressor class')
    parser.add_argument('--regressor-hyperparameters', default=None, help='YAML file of keyword arguments for the regressor class (default: None, use empty dict)')
    parser.add_argument('--window-slider', default='pyseq.models.window_sliders.window_slide.UnivariateWindowSlider', help='Dotted path to the window slider class (default: WindowSlider)')
    parser.add_argument('--window-slider-kwargs', default=None, help='YAML file of keyword arguments for the window slider class (default: None, use empty dict)')
    parser.add_argument('--thresholder', default='pyseq.models.thresholders.wald_constant_thresholder.WaldConstantThresholder', help='Dotted path to the thresholder class (default: WaldConstantThresholder)')
    parser.add_argument('--thresholder-kwargs', default=None, help='YAML file of keyword arguments for the thresholder class (default: None, use empty dict)')
    parser.add_argument('--scorer', default='pyseq.models.scorers.cusum.BidirectionalCUSUMScorer', help='Score function to convert regression output to scores (default: cusum)')
    parser.add_argument('--scorer-kwargs', default=None, help='YAML file of keyword arguments for the scorer class (default: None, use empty dict)')
    parser.add_argument('--plot-test-results', action='store_true', help='Whether to plot test results (default: False)')
    parser.add_argument('--write-results', action='store_true', help='Whether to plot test results (default: False)')

    args = parser.parse_args()

    results_path = os.path.dirname(__file__)+"/results/"
    os.makedirs(results_path + "/figures/", exist_ok=True)

    # Use the generator and model
    # Parse model and generator kwargs from YAML 
    # If the files are not provided, use empty dicts as kwargs
    # Load regressor, window-slider, and thresholder hyperparameters
    regressor_kwargs = handle_open_file(args.regressor_hyperparameters)
    window_slider_kwargs = handle_open_file(args.window_slider_kwargs)
    thresholder_kwargs = handle_open_file(args.thresholder_kwargs)
    scorer_kwargs = handle_open_file(args.scorer_kwargs)
    
    # Import window slider, regressor, thresholder, scorer
    regressor_cls = import_object_from_string(args.regressor)
    regressor = regressor_cls(**regressor_kwargs)
    scorer_cls = import_object_from_string(args.scorer)
    scorer = scorer_cls(**scorer_kwargs)
    window_slider_cls = import_object_from_string(args.window_slider)
    window_slider = window_slider_cls(**window_slider_kwargs)
    thresholder_cls = import_object_from_string(args.thresholder)
    thresholder = thresholder_cls(**thresholder_kwargs)

    sd = StackDetector(window_slider=window_slider, regressor=regressor, 
                       thresholder=thresholder, 
                       scorer=scorer)

    t_train, y_train, cps, dataset_name = make_dataset(generator_hyperparameters=args.generator_hyperparameters, generator_name="test", set_name="train")

    # Training (only if model is fittable)
    if regressor.fittable:
        print("Model is fittable, training...")
        # If we want to add preprocessing steps, add them to this function call
        sd.fit(y_s=y_train, cps_s=cps)

    # Testing
    print("Testing phase...")
    pred_test, scores_test, reg_pred_test = sd.predict(y_s=y_train, return_scores=True, return_regressor_predictions=True)

    # optional test plotting:
    if args.plot_test_results:
        plot_cusum_results([t_train, y_train, cps, pred_test, scores_test, reg_pred_test], sd.thresholder.alpha, filename=results_path + "/figures/" + dataset_name + "_")

    if args.write_results:
        write_results(results_path + dataset_name, [pred_test, scores_test, reg_pred_test])


# For testing purposes, provide defaults if not running as a script
if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--generator-hyperparameters", os.path.join(SCRIPT_DIR, "data_config.yaml"),
        "--regressor", "pyseq.models.regressors.random_forest_regression.AutoRegressiveRandomForest",
        "--window-slider-kwargs", os.path.join(SEQBENCH_DIR, "config", "window_slider.yaml"),
        "--thresholder-kwargs", os.path.join(SEQBENCH_DIR, "config", "wald-constant-thresholder.yaml"),
        "--scorer-kwargs", os.path.join(SEQBENCH_DIR, "config", "cusum.yaml"),
        "--plot-test-results",
        "--write-results"
    ]
    main()
