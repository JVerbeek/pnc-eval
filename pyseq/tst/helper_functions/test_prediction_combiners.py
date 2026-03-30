import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.prediction_combiners.prediction_combiners import select_first, select_last, select_mean

class TestSelectFirst:
    def test_non_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        window_indices = [(0, 3), (3, 6)]
        result = select_first(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_overlapping_windows_selects_first(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]
        window_indices = [(0, 4), (2, 6)]
        result = select_first(window_predictions, window_indices)
        # Overlap at indices 2,3: first window's values (3.0, 4.0) should be selected
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 30.0, 40.0])

    def test_single_window(self):
        window_predictions = [np.array([5.0, 6.0, 7.0])]
        window_indices = [(0, 3)]
        result = select_first(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [5.0, 6.0, 7.0])

    def test_fully_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
        ]
        window_indices = [(0, 3), (0, 3)]
        result = select_first(window_predictions, window_indices)
        # First prediction should win
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_three_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([100.0, 200.0, 300.0]),
        ]
        window_indices = [(0, 3), (1, 4), (2, 5)]
        result = select_first(window_predictions, window_indices)
        # Index 0: only window 0 -> 1.0
        # Index 1: window 0 and 1 -> first = 2.0
        # Index 2: all three windows -> first = 3.0
        # Index 3: window 1 and 2 -> first = 20.0
        # Index 4: only window 2 -> 300.0
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 30.0, 300.0])

    def test_window_not_starting_at_zero(self):
        window_predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        window_indices = [(2, 4), (4, 6)]
        result = select_first(window_predictions, window_indices)
        # Indices 0,1 should be zeros (initialized), 2-3 from first, 4-5 from second
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0])


class TestSelectLast:
    def test_non_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        window_indices = [(0, 3), (3, 6)]
        result = select_last(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_overlapping_windows_selects_last(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]
        window_indices = [(0, 4), (2, 6)]
        result = select_last(window_predictions, window_indices)
        # Overlap at indices 2,3: last window's values (10.0, 20.0) should be selected
        np.testing.assert_array_equal(result, [1.0, 2.0, 10.0, 20.0, 30.0, 40.0])

    def test_single_window(self):
        window_predictions = [np.array([5.0, 6.0, 7.0])]
        window_indices = [(0, 3)]
        result = select_last(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [5.0, 6.0, 7.0])

    def test_fully_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
        ]
        window_indices = [(0, 3), (0, 3)]
        result = select_last(window_predictions, window_indices)
        # Last prediction should win
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_three_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([100.0, 200.0, 300.0]),
        ]
        window_indices = [(0, 3), (1, 4), (2, 5)]
        result = select_last(window_predictions, window_indices)
        # Index 0: only window 0 -> 1.0
        # Index 1: window 0 and 1 -> last = 10.0
        # Index 2: all three windows -> last = 100.0
        # Index 3: window 1 and 2 -> last = 200.0
        # Index 4: only window 2 -> 300.0
        np.testing.assert_array_equal(result, [1.0, 10.0, 100.0, 200.0, 300.0])

    def test_window_not_starting_at_zero(self):
        window_predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        window_indices = [(2, 4), (4, 6)]
        result = select_last(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0])


class TestSelectMean:
    def test_non_overlapping_windows(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        window_indices = [(0, 3), (3, 6)]
        result = select_mean(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_overlapping_windows_computes_mean(self):
        window_predictions = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]
        window_indices = [(0, 4), (2, 6)]
        result = select_mean(window_predictions, window_indices)
        # Index 0: 1.0, Index 1: 2.0
        # Index 2: mean(3.0, 10.0) = 6.5
        # Index 3: mean(4.0, 20.0) = 12.0
        # Index 4: 30.0, Index 5: 40.0
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 6.5, 12.0, 30.0, 40.0])

    def test_single_window(self):
        window_predictions = [np.array([5.0, 6.0, 7.0])]
        window_indices = [(0, 3)]
        result = select_mean(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [5.0, 6.0, 7.0])

    def test_fully_overlapping_windows(self):
        window_predictions = [
            np.array([2.0, 4.0, 6.0]),
            np.array([10.0, 20.0, 30.0]),
        ]
        window_indices = [(0, 3), (0, 3)]
        result = select_mean(window_predictions, window_indices)
        np.testing.assert_array_almost_equal(result, [6.0, 12.0, 18.0])

    def test_three_overlapping_windows(self):
        window_predictions = [
            np.array([3.0, 6.0, 9.0]),
            np.array([12.0, 15.0, 18.0]),
            np.array([21.0, 24.0, 27.0]),
        ]
        window_indices = [(0, 3), (1, 4), (2, 5)]
        result = select_mean(window_predictions, window_indices)
        # Index 0: 3.0 (count=1)
        # Index 1: (6.0 + 12.0) / 2 = 9.0
        # Index 2: (9.0 + 15.0 + 21.0) / 3 = 15.0
        # Index 3: (18.0 + 24.0) / 2 = 21.0
        # Index 4: 27.0 (count=1)
        np.testing.assert_array_almost_equal(result, [3.0, 9.0, 15.0, 21.0, 27.0])

    def test_no_division_by_zero_for_uncovered_indices(self):
        window_predictions = [
            np.array([1.0, 2.0]),
        ]
        window_indices = [(2, 4)]
        result = select_mean(window_predictions, window_indices)
        # Indices 0, 1 are uncovered and should remain 0.0
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0])

    def test_window_not_starting_at_zero(self):
        window_predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        window_indices = [(2, 4), (4, 6)]
        result = select_mean(window_predictions, window_indices)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0])

    def test_mean_with_many_overlapping_windows(self):
        window_predictions = [
            np.array([10.0]),
            np.array([20.0]),
            np.array([30.0]),
            np.array([40.0]),
        ]
        window_indices = [(0, 1), (0, 1), (0, 1), (0, 1)]
        result = select_mean(window_predictions, window_indices)
        np.testing.assert_array_almost_equal(result, [25.0])


class TestEdgeCases:
    def test_select_first_with_float_predictions(self):
        window_predictions = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        ]
        window_indices = [(0, 2), (1, 3)]
        result = select_first(window_predictions, window_indices)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.4])

    def test_select_last_with_float_predictions(self):
        window_predictions = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        ]
        window_indices = [(0, 2), (1, 3)]
        result = select_last(window_predictions, window_indices)
        np.testing.assert_array_almost_equal(result, [0.1, 0.3, 0.4])

    def test_select_mean_with_negative_values(self):
        window_predictions = [
            np.array([-1.0, -2.0, -3.0]),
            np.array([1.0, 2.0, 3.0]),
        ]
        window_indices = [(0, 3), (0, 3)]
        result = select_mean(window_predictions, window_indices)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_large_window_size(self):
        size = 10000
        window_predictions = [np.ones(size), np.ones(size) * 2]
        window_indices = [(0, size), (0, size)]
        result_mean = select_mean(window_predictions, window_indices)
        np.testing.assert_array_almost_equal(result_mean, np.ones(size) * 1.5)

    def test_consistency_no_overlap(self):
        """When there's no overlap, all three combiners should return the same result."""
        window_predictions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ]
        window_indices = [(0, 2), (2, 4), (4, 6)]
        result_first = select_first(window_predictions, window_indices)
        result_last = select_last(window_predictions, window_indices)
        result_mean = select_mean(window_predictions, window_indices)
        np.testing.assert_array_equal(result_first, result_last)
        np.testing.assert_array_equal(result_first, result_mean)


#manual tests for now, but should be converted to proper unit tests with pytest
if __name__ == "__main__":
    test_select_first = TestSelectFirst()
    test_select_first.test_non_overlapping_windows()
    test_select_first.test_overlapping_windows_selects_first()
    test_select_first.test_single_window()
    test_select_first.test_fully_overlapping_windows()
    test_select_first.test_three_overlapping_windows()
    test_select_first.test_window_not_starting_at_zero()

    test_select_last = TestSelectLast()
    test_select_last.test_non_overlapping_windows()
    test_select_last.test_overlapping_windows_selects_last()
    test_select_last.test_single_window()
    test_select_last.test_fully_overlapping_windows()
    test_select_last.test_three_overlapping_windows()
    test_select_last.test_window_not_starting_at_zero()

    test_select_mean = TestSelectMean()
    test_select_mean.test_non_overlapping_windows()
    test_select_mean.test_overlapping_windows_computes_mean()
    test_select_mean.test_single_window()
    test_select_mean.test_fully_overlapping_windows()
    test_select_mean.test_three_overlapping_windows()
    test_select_mean.test_no_division_by_zero_for_uncovered_indices()
    test_select_mean.test_window_not_starting_at_zero()
    test_select_mean.test_mean_with_many_overlapping_windows()


    test_edge_cases = TestEdgeCases()
    test_edge_cases.test_select_first_with_float_predictions()
    test_edge_cases.test_select_last_with_float_predictions()
    test_edge_cases.test_select_mean_with_negative_values()
    test_edge_cases.test_large_window_size()
    test_edge_cases.test_consistency_no_overlap()

    print("All tests passed!")
    