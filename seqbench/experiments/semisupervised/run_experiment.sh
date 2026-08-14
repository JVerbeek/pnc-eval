python3 ../../masterscript.py --generator-hyperparameters experiment-config/data_config.yaml \
    --window-slider-kwargs ../../config/window_slider.yaml\
    --regressor pyseq.models.regressors.random_forest_regression.MultiOutputRandomForest\
    --thresholder-kwargs experiment-config/wald-constant-thresholder.yaml\
    --scorer-kwargs experiment-config/cusum.yaml\
    --experiment-name "semisupervised"\
    --write-results
