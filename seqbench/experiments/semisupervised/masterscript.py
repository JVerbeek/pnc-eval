import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # this is moved up two dirs, so effectively
SEQBENCH_DIR = os.path.dirname(
    os.path.dirname(SCRIPT_DIR)
)  # this is the same dir, but then the experiment name becomes more important.
sys.path.append(SEQBENCH_DIR)
import matplotlib.pyplot as plt
import numpy as np
from utils import handle_open_file, import_object_from_string, write_results
from plotting import plot_cusum_results

from pyseq.models.base_stack_detector import StackDetector
from run_dataset_generation import make_dataset


def generate_simple_mean_changes(path_to_data_config):
    t_test, y_test, cps = make_dataset(
        generator_hyperparameters=path_to_data_config,
        generator_name="simple_interpretable",
        set_name="simple_interpretable",
    )
    return t_test, y_test, cps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-hyperparameters", default=None)
    parser.add_argument(
        "--regressor", required=True, help="Dotted path to the regressor class"
    )
    parser.add_argument(
        "--regressor-hyperparameters",
        default=None,
        help="YAML file of keyword arguments for the regressor class (default: None, use empty dict)",
    )
    parser.add_argument(
        "--window-slider",
        default="pyseq.models.window_sliders.window_slide.UnivariateWindowSlider",
        help="Dotted path to the window slider class, or 'none' to run without one (default: WindowSlider). 'none' is only valid for streaming regressors, which do their own windowing.",
    )
    parser.add_argument(
        "--window-slider-kwargs",
        default=None,
        help="YAML file of keyword arguments for the window slider class (default: None, use empty dict)",
    )
    parser.add_argument(
        "--predictor-window-size",
        type=int,
        default=None,
        help="Only with --window-slider none: how much history each prediction is made from. For a streaming model such as an LSTM this is only the warm-up period, since its carried state sees the whole series regardless; setting it to the same value a windowed method would use keeps the first prediction at the same index for both, which is what makes the comparison fair. Passed to the regressor, which is then the only source of window geometry.",
    )
    parser.add_argument(
        "--skip-length",
        type=int,
        default=None,
        help="Only with --window-slider none: how far the window advances between consecutive predictions. Passed to the regressor, which is then the only source of window geometry.",
    )
    parser.add_argument(
        "--target-window-size",
        type=int,
        default=None,
        help="Only with --window-slider none: how many steps ahead each prediction covers. Passed to the regressor, which is then the only source of window geometry.",
    )
    parser.add_argument(
        "--thresholder",
        default="pyseq.models.thresholders.wald_constant_thresholder.WaldConstantThresholder",
        help="Dotted path to the thresholder class (default: WaldConstantThresholder)",
    )
    parser.add_argument(
        "--thresholder-kwargs",
        default=None,
        help="YAML file of keyword arguments for the thresholder class (default: None, use empty dict)",
    )
    parser.add_argument(
        "--scorer",
        default="pyseq.models.scorers.cusum.BidirectionalCUSUMScorer",
        help="Score function to convert regression output to scores (default: cusum)",
    )
    parser.add_argument(
        "--scorer-kwargs",
        default=None,
        help="YAML file of keyword arguments for the scorer class (default: None, use empty dict)",
    )
    parser.add_argument(
        "--plot-test-results",
        action="store_true",
        help="Whether to plot test results (default: False)",
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Whether to plot test results (default: False)",
    )

    args = parser.parse_args()

    results_path = os.path.dirname(__file__) + "/results/"
    os.makedirs(results_path + "/figures/", exist_ok=True)

    # Use the generator and model
    # Parse model and generator kwargs from YAML
    # If the files are not provided, use empty dicts as kwargs
    # Load regressor, window-slider, and thresholder hyperparameters
    regressor_kwargs = handle_open_file(args.regressor_hyperparameters)
    window_slider_kwargs = handle_open_file(args.window_slider_kwargs)
    thresholder_kwargs = handle_open_file(args.thresholder_kwargs)
    scorer_kwargs = handle_open_file(args.scorer_kwargs)

    # Build the window slider, or skip it entirely. Without a slider the regressor is the only
    # source of window geometry, so the three geometry settings the slider would have carried
    # have to be set by hand and handed to the regressor instead.
    regressor_cls = import_object_from_string(args.regressor)
    no_window_slider = (
        args.window_slider is None or args.window_slider.lower() == "none"
    )

    manual_geometry = {
        "predictor_window_size": args.predictor_window_size,
        "skip_length": args.skip_length,
        "target_window_size": args.target_window_size,
    }

    if no_window_slider:
        # Checked here rather than left to the StackDetector, so that the geometry is never
        # forwarded as kwargs to a regressor that has no idea what to do with it.
        if getattr(regressor_cls, "processing", None) != "streaming":
            parser.error(
                f"--window-slider none is only valid for streaming regressors; {regressor_cls.__name__} needs a window slider to cut its input windows."
            )

        missing = sorted(
            name for name, value in manual_geometry.items() if value is None
        )
        if missing:
            parser.error(
                f"{', '.join('--' + name.replace('_', '-') for name in missing)} must be given when --window-slider is 'none', as there is no window slider to take the geometry from."
            )
        if window_slider_kwargs:
            parser.error(
                "--window-slider-kwargs cannot be combined with --window-slider none; there is no window slider to configure."
            )

        window_slider = None
        regressor_kwargs = {**regressor_kwargs, **manual_geometry}
    else:
        supplied = sorted(
            name for name, value in manual_geometry.items() if value is not None
        )
        if supplied:
            parser.error(
                f"{', '.join('--' + name.replace('_', '-') for name in supplied)} {'is' if len(supplied) == 1 else 'are'} only used with --window-slider none; set the geometry in --window-slider-kwargs instead."
            )

        window_slider_cls = import_object_from_string(args.window_slider)
        window_slider = window_slider_cls(**window_slider_kwargs)

    # Import regressor, thresholder, scorer
    regressor = regressor_cls(**regressor_kwargs)
    scorer_cls = import_object_from_string(args.scorer)
    scorer = scorer_cls(**scorer_kwargs)
    thresholder_cls = import_object_from_string(args.thresholder)
    thresholder = thresholder_cls(**thresholder_kwargs)

    sd = StackDetector(
        window_slider=window_slider,
        regressor=regressor,
        thresholder=thresholder,
        scorer=scorer,
    )

    t_train, y_train, cps, dataset_name = make_dataset(
        generator_hyperparameters=args.generator_hyperparameters,
        generator_name="test",
        set_name="train",
    )

    # Training (only if model is fittable)
    if regressor.fittable:
        print("Model is fittable, training...")
        # If we want to add preprocessing steps, add them to this function call
        sd.fit(y_s=y_train, cps_s=cps)

    # Testing
    print("Testing phase...")
    pred_test, scores_test, reg_pred_test = sd.predict(
        y_s=y_train, return_scores=True, return_regressor_predictions=True
    )

    # optional test plotting:
    if args.plot_test_results:
        plot_cusum_results(
            [t_train, y_train, cps, pred_test, scores_test, reg_pred_test],
            sd.thresholder.alpha,
            filename=results_path + "/figures/" + dataset_name + "_",
        )

    if args.write_results:
        write_results(
            results_path + dataset_name, [pred_test, scores_test, reg_pred_test]
        )


# For testing purposes, provide defaults if not running as a script.
# Keyed on --regressor rather than on argv being empty: --regressor is required, so its presence
# is what distinguishes a real command line (which must then win, so that options like
# --window-slider none are reachable) from a bare run. Launchers that inject argv of their own --
# the VS Code interactive window's --f=..., debugpy, pytest -- have no --regressor, so they still
# get the defaults, and replacing argv wholesale drops their stray arguments with it.
if __name__ == "__main__":
    if not any(
        arg == "--regressor" or arg.startswith("--regressor=") for arg in sys.argv[1:]
    ):
        sys.argv = [
            sys.argv[0],
            "--generator-hyperparameters",
            os.path.join(SCRIPT_DIR, "data_config.yaml"),
            # "--regressor", "pyseq.models.regressors.random_forest_regression.AutoRegressiveRandomForest",
            # "--regressor", "pyseq.models.regressors.xgboost_regression.AutoRegressiveXGBoost",
            # "--regressor", "pyseq.models.regressors.xgboost_regression.MultiOutputXGBoost",
            "--regressor",
            "pyseq.models.regressors.lstm_regression.MultiOutputLSTM",
            # "--regressor-hyperparameters", os.path.join(SEQBENCH_DIR, "config", "models", "lstm-no-standardize.yaml"),
            # "--window-slider-kwargs", os.path.join(SEQBENCH_DIR, "config", "window_slider.yaml"),
            # Slider-less alternative for streaming regressors (swap in for the line above).
            # Mirrors config/window_slider.yaml, so the comparison against the windowed methods
            # stays like for like:
            "--window-slider",
            "none",
            "--predictor-window-size",
            "50",
            "--skip-length",
            "1",
            "--target-window-size",
            "1",
            "--thresholder-kwargs",
            os.path.join(SEQBENCH_DIR, "config", "wald-constant-thresholder.yaml"),
            "--scorer-kwargs",
            os.path.join(SEQBENCH_DIR, "config", "cusum.yaml"),
            "--plot-test-results",
            "--write-results",
        ]
    main()
