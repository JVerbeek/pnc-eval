import numpy as np 

import abc

class Slider():   
    def __init__(self, window_size, skip_length):
        self.window_size = window_size
        self.skip_length = skip_length

    @abc.abstractmethod
    def new_slide(self, X):
        pass

    @abc.abstractmethod
    def next_window(self):
        pass

class WindowSlider(Slider):
    def __init__(self, window_size, skip_length):
        super().__init__(window_size, skip_length)
        self.current_position = 0
        self.X = None

        # Are used doubly with current_position, but this is hopefully more legible
        self.window_start_index = None
        self.window_end_index = None

    def new_slide(self, X):
        self.X = X
        self.current_position = 0

    def next_window(self):
        if self.X is None:
            raise ValueError("No data provided. Please call new_slide(X) before next_window().")
        n = len(self.X)
        while self.current_position + self.window_size <= n:
            window = self.X[self.current_position:self.current_position + self.window_size]
            self.window_start_index = self.current_position
            self.window_end_index = self.current_position + self.window_size
            self.current_position += self.skip_length
            
            yield window

        # Yield the last, incomplete window if any data remains
        if self.current_position < n:
            window = self.X[self.current_position:n]
            self.window_start_index = self.current_position
            self.window_end_index = n
            self.current_position = n  # Move to end
            yield window

    def get_window_indices(self):
        if self.X is None:
            raise ValueError("No data provided. Please call new_slide(X) before get_window_indices().")
        
        if self.window_start_index is None or self.window_end_index is None:
            raise ValueError("No window has been generated yet. Please call next_window() before get_window_indices().")
        
        return self.window_start_index, self.window_end_index

        

class NonOverlappingWindowSlider(WindowSlider):
    def __init__(self, window_size):
        super().__init__(window_size, window_size)