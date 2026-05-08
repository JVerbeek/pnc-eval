import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.window_sliders.window_slide import UnivariateWindowSlider

import numpy as np

def test_univariate_window_slider_initialization():
    slider = UnivariateWindowSlider(predictor_window_size=5, skip_length=2, target_window_size=3)
    assert slider.predictor_window_size == 5
    assert slider.skip_length == 2
    assert slider.target_window_size == 3

def test_univariate_window_slider_window_extraction():
    data = np.arange(20)  # Sample data from 0 to 19
    slider = UnivariateWindowSlider(predictor_window_size=5, skip_length=3, target_window_size=2)
    slider.new_slide(data)

    expected_predictor_windows = [
        np.array([0, 1, 2, 3, 4]),
        np.array([3, 4, 5, 6, 7]),
        np.array([6, 7, 8, 9, 10]),
        np.array([9, 10, 11, 12, 13]),
        np.array([12, 13, 14, 15, 16])
    ]

    expected_target_windows = [
        np.array([5, 6]),
        np.array([8, 9]),
        np.array([11, 12]),
        np.array([14, 15]),
        np.array([17, 18])
    ]

    for i, (predictor_window, target_window) in enumerate(slider):
        np.testing.assert_array_equal(predictor_window, expected_predictor_windows[i])
        np.testing.assert_array_equal(target_window, expected_target_windows[i])

def test_univariate_window_slider_get_all_windows():
    data = np.arange(15)  # Sample data from 0 to 14
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=2, target_window_size=2)
    slider.new_slide(data)

    expected_predictor_windows = np.array([
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
        [8, 9, 10, 11]
    ])

    expected_target_windows = np.array([
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13]
    ])

    predictor_windows, target_windows = slider.get_all_windows()

    np.testing.assert_array_equal(predictor_windows, expected_predictor_windows)
    np.testing.assert_array_equal(target_windows, expected_target_windows)

def test_2d_input():
    n = 15
    data = np.arange(n).reshape(n,1)
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=1, target_window_size=2)
    
    slider.new_slide(data)

    windows = list(slider)
    expected_num_windows = n - slider.predictor_window_size - slider.target_window_size + 1
    assert len(windows) == expected_num_windows

    data = np.arange(n).reshape(1, n)
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=1, target_window_size=2)
    
    slider.new_slide(data)

    windows = list(slider)
    expected_num_windows = n - slider.predictor_window_size - slider.target_window_size + 1
    assert len(windows) == expected_num_windows


def test_insufficient_data():
    data = np.arange(5)  # Sample data from 0 to 4
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=1, target_window_size=2)
    slider.new_slide(data)

    windows = list(slider)
    assert len(windows) == 0

def test_no_data():
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=1, target_window_size=2)
    try:
        slider.get_all_windows()
        assert False, "Expected ValueError when calling get_all_windows() without new_slide()"
    except ValueError as e:
        assert str(e) == "No data provided. Please call new_slide(y) before get_all_windows()."

def test_next_window_no_data():
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=1, target_window_size=2)
    try:
        slider.next_window()
        assert False, "Expected ValueError when calling next_window() without new_slide()"
    except ValueError as e:
        assert str(e) == "No data provided. Please call new_slide(y) before next_window()."

def test_non_univariate_input():
    slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=1, target_window_size=2)
    data = np.array([[1, 2], [3, 4], [5, 6]])
    try:
        slider.new_slide(data)
        assert False, "Expected ValueError when calling new_slide() with non-univariate input"
    except ValueError as e:
        assert str(e) == "Input data y must be univariate (1D array)."

def test_skip_length_larger_than_prediction_window_size():
    try:
        slider = UnivariateWindowSlider(predictor_window_size=4, skip_length=5, target_window_size=2)
        assert slider.skip_length == 5
    except ValueError as e:
        assert str(e) == "prediction_window_size must be at least as large as the window_slider's skip_length."

def test_return_indices():
    data = np.arange(20)  # Sample data from 0 to 19
    slider = UnivariateWindowSlider(predictor_window_size=5, skip_length=3, target_window_size=2)
    slider.new_slide(data)

    expected_indices = [
        (0, 5),
        (3, 8),
        (6, 11),
        (9, 14),
        (12, 17)
    ]

    for i, ((_, _), indices) in enumerate(slider.next_window(return_indices=True)):
        assert indices == expected_indices[i]

def test_t_input():
    data = np.arange(20)  # Sample data from 0 to 19
    time = np.arange(20) * 0.5  # Time component, should be ignored
    slider = UnivariateWindowSlider(predictor_window_size=5, skip_length=3, target_window_size=2)
    try:
        slider.new_slide(data, t=time)
        assert False, "Expected UserWarning when calling new_slide() with time input"
    except UserWarning as e:
        assert str(e) == "Time component t is provided but will be ignored for UnivariateWindowSlider."

def test_X_input():
    data = np.arange(20)  # Sample data from 0 to 19
    exogenous_predictors = np.arange(20).reshape(-1, 1)  # Predictor component, should be ignored
    slider = UnivariateWindowSlider(predictor_window_size=5, skip_length=3, target_window_size=2)
    try:
        slider.new_slide(data, X=exogenous_predictors)
        assert False, "Expected UserWarning when calling new_slide() with predictor input"
    except UserWarning as e:
        assert str(e) == "Predictor component X is provided but will be ignored for UnivariateWindowSlider."

#temporary test calls to run script manually:

if __name__ == "__main__":
    test_univariate_window_slider_initialization()
    test_univariate_window_slider_window_extraction()
    test_univariate_window_slider_get_all_windows()
    test_2d_input()
    test_insufficient_data()
    test_no_data()
    test_next_window_no_data()
    test_non_univariate_input()
    test_skip_length_larger_than_prediction_window_size()
    test_return_indices()
    test_t_input()
    test_X_input()
    print("All tests passed!")