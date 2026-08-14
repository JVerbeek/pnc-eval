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