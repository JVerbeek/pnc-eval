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


