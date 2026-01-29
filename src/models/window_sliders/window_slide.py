import numpy as np 

import abc

class Slider(abc.ABC):   
    def __init__(self, predictor_window_size, skip_length = 1, target_window_size=1):
        #Check if predictor_window_size and skip_length are positive integers
        if not (isinstance(predictor_window_size, int) and predictor_window_size > 0):
            raise ValueError("predictor_window_size must be a positive integer.")
        if not (isinstance(skip_length, int) and skip_length > 0):
            raise ValueError("skip_length must be a positive integer.")
        if not (isinstance(target_window_size, int) and target_window_size > 0):
            raise ValueError("target_window_size must be a positive integer.")

        self.predictor_window_size = predictor_window_size
        self.target_window_size = target_window_size
        self.skip_length = skip_length


    def new_slide(self, y, t=None, X=None):
        self.y = y
        self.t = t
        self.X = X

        self.current_position = 0

    @abc.abstractmethod
    def next_window(self):
        pass

    def __iter__(self):
        return self.next_window()

    def get_all_windows(self):
        #calculate maximum number of windows
        if self.y is None:
            raise ValueError("No data provided. Please call new_slide(y) before get_all_windows().")
            
        num_windows = (len(self.y) - self.predictor_window_size - self.target_window_size) // self.skip_length + 1
        if num_windows < 0:
            num_windows = 0

        predictor_windows = np.empty((num_windows, self.predictor_window_size))
        target_windows = np.empty((num_windows, self.target_window_size))

        for i, (predictor_window, target_window) in enumerate(self):
            predictor_windows[i] = predictor_window
            target_windows[i] = target_window

        return predictor_windows, target_windows

class UnivariateWindowSlider(Slider):
    def __init__(self, predictor_window_size, skip_length=1, target_window_size=1):
        super().__init__(predictor_window_size, skip_length, target_window_size)
        self.current_position = 0
        self.y = None

    def new_slide(self, y):
        #make sure y is 1D, drop dimensions if needed
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError("Input data y must be univariate (1D array).")
        
        super().new_slide(y)

    def next_window(self, return_indices=False):
        # return indices returns predictor window indices
        if self.y is None:
            raise ValueError("No data provided. Please call new_slide(y) before next_window().")

        n = len(self.y)

        while self.current_position + self.predictor_window_size + self.target_window_size <= n:
            predictor_window = self.y[self.current_position:self.current_position + self.predictor_window_size]
            target_window = self.y[self.current_position + self.predictor_window_size:self.current_position + self.predictor_window_size + self.target_window_size]
            self.current_position += self.skip_length

            if return_indices:
                predictor_window_start_index = self.current_position - self.skip_length
                predictor_window_end_index = predictor_window_start_index + self.predictor_window_size
                yield (predictor_window, target_window), (predictor_window_start_index, predictor_window_end_index)
            else:
                yield predictor_window, target_window



# class WindowSlider(Slider):
#     def __init__(self, window_size, skip_length):
#         super().__init__(window_size, skip_length)
#         self.current_position = 0
#         self.X = None

#         # Are used doubly with current_position, but this is hopefully more legible
#         self.window_start_index = None
#         self.window_end_index = None

#     def next_window(self):
#         if self.X is None:
#             raise ValueError("No data provided. Please call new_slide(X) before next_window().")
#         n = len(self.X)
#         while self.current_position + self.window_size <= n:
#             window = self.X[self.current_position:self.current_position + self.window_size]
#             self.window_start_index = self.current_position
#             self.window_end_index = self.current_position + self.window_size
#             self.current_position += self.skip_length
            
#             yield window

#         # Yield the last, incomplete window if any data remains
#         if self.current_position < n:
#             window = self.X[self.current_position:n]
#             self.window_start_index = self.current_position
#             self.window_end_index = n
#             self.current_position = n  # Move to end
#             yield window

#     def get_window_indices(self):
#         if self.X is None:
#             raise ValueError("No data provided. Please call new_slide(X) before get_window_indices().")
        
#         if self.window_start_index is None or self.window_end_index is None:
#             raise ValueError("No window has been generated yet. Please call next_window() before get_window_indices().")
        
#         return self.window_start_index, self.window_end_index

# class NonOverlappingWindowSlider(WindowSlider):
#     def __init__(self, window_size):
#         super().__init__(window_size, window_size)

