import numpy as np
#import pytest
from src.helper_functions.data_transformation import transform_for_autoregressive_fit

def test_transform_basic():
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    y = np.array([[10, 11],
                  [20, 21]])
    X_ar, y_ar = transform_for_autoregressive_fit(X, y)

    expected_X_ar = np.array([
        [1, 2, 3],   # i=0, j=0
        [2, 3, 10],  # i=0, j=1
        [4, 5, 6],   # i=1, j=0
        [5, 6, 20],  # i=1, j=1
    ])
    expected_y_ar = np.array([10, 11, 20, 21])

    np.testing.assert_array_equal(X_ar, expected_X_ar)
    np.testing.assert_array_equal(y_ar, expected_y_ar)
    assert X_ar.shape == (4, 3)
    assert y_ar.shape == (4,)

def test_transform_prediction_equals_one():
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    y = np.array([[10],
                  [20]])
    X_ar, y_ar = transform_for_autoregressive_fit(X, y)

    expected_X_ar = np.array([
        [1, 2, 3],  # i=0, j=0
        [4, 5, 6],  # i=1, j=0
    ])
    expected_y_ar = np.array([10, 20])

    np.testing.assert_array_equal(X_ar, expected_X_ar)
    np.testing.assert_array_equal(y_ar, expected_y_ar)
    assert X_ar.shape == (2, 3)
    assert y_ar.shape == (2,)

def test_transform_prediction_equals_window():
    X = np.array([[1, 2, 3]])
    y = np.array([[10, 11, 12]])
    X_ar, y_ar = transform_for_autoregressive_fit(X, y)

    expected_X_ar = np.array([
        [1, 2, 3],    # j=0
        [2, 3, 10],   # j=1
        [3, 10, 11],  # j=2
    ])
    expected_y_ar = np.array([10, 11, 12])

    np.testing.assert_array_equal(X_ar, expected_X_ar)
    np.testing.assert_array_equal(y_ar, expected_y_ar)
    assert X_ar.shape == (3, 3)
    assert y_ar.shape == (3,)

def test_empty_input_windows():
    # zero windows should produce empty outputs with correct shapes
    X = np.zeros((0, 4))
    y = np.zeros((0, 2))
    X_ar, y_ar = transform_for_autoregressive_fit(X, y)

    assert X_ar.shape == (0, 4)
    assert y_ar.size == 0
    # ensure arrays are numpy arrays
    assert isinstance(X_ar, np.ndarray)
    assert isinstance(y_ar, np.ndarray)