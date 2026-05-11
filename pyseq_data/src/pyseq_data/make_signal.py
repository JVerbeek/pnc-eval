from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt  

@dataclass 
class Property:
    def __init__(self, value, dist=None):
        self.value = value
        self.distribution = dist 

    def asarray(self, length):
        arr =  np.array([self.value] * length)
        return arr

    def get_value(self):
        return self.value

    def update_value(self):
        self.value = self.distribution.rvs(1) 

@dataclass
class DataType:
    mean: Property
    variance: Property
    time_start: float
    time_stop: float 
    length: int

    def __post_init__(self):
        for fname, ftype in self.__annotations__.items():
            f = getattr(self, fname)
            if isinstance(f, Property):
                f.value = f.asarray(self.length)

    def get_data(self, n_datasets):
        ts, ys = np.zeros((n_datasets, self.length)), np.zeros((n_datasets, self.length))
        for i in range(n_datasets):
            t, y = self.get_single_dataset()
            ts[i] = t
            ys[i] = y
            self.update_params()
        return ts, ys, np.array([self.location] * n_datasets)

    def update_params(self):
        for key, prop in vars(self).items():
            if isinstance(prop, Property):
                prop.update_value()
            if isinstance(prop, ChangeDecorator):
                prop.before_change.update_value()
                after_change = prop.before_change.distribution.rvs(1)

@dataclass
class ChangeDecorator():
    def __init__(self, location=0, before_change=Property(0), after_change=Property(1), length=100):
        self.location = location 
        self.before_change = before_change
        self.after_change = after_change
        self.length = length

    def get_value(self):
        v = self.value
        return v.flatten()
        
@dataclass
class PolyChange(ChangeDecorator):
    def __init__(self, order=1, location=0, before_change=Property(0), after_change=1, length=100):
        super().__init__(location, before_change, after_change, length)
        self.order = order
        if order == 0:
            raise UserWarning("Order of PolyChange is set to 0. This will not generate a change.")

    def get_value(self):
        t = np.arange(self.length)
        t_prime = self.get_fn()(t)
        return t_prime.flatten()

    def get_fn(self):
        return lambda t: np.where(t < self.location, self.before_change.get_value(), self.before_change.get_value() + t ** self.order * ((self.after_change - self.before_change.get_value()) / (max(t[self.location:]) - min(t[self.location:]))))

@dataclass
class SteppedChange(ChangeDecorator):
    def __init__(self, location=0, before_change=Property(0), after_change=1, length=100):
        super().__init__(location, before_change, after_change, length)

    def get_value(self):
        t = np.arange(self.length)
        t_prime = self.get_fn()(t)
        return t_prime.flatten()


    def get_fn(self):
        return lambda t: np.where(t < self.location, self.before_change.value, self.after_change)

@dataclass
class ConstantData(DataType):
    mean: Property
    variance: Property
    location: 0

    def get_single_dataset(self):
        t = np.linspace(self.time_start, self.time_stop, self.length)
        mean = self.mean.get_value().flatten()
        std = self.variance.get_value().flatten()
        y = mean + np.random.normal(0, std**2, self.length) * std**2
        y = y.flatten()
        if not isinstance(self.mean, SteppedChange):
            y[:self.location] += y[self.location:][0] - y[:self.location][-1]
        y[:self.location] -= y[self.location:][0] - y[:self.location][-1]
        return t, y 

@dataclass
class OscillationData(DataType):
    mean: Property
    variance: Property
    amplitude: Property
    frequency: Property
    location: 0

    def get_single_dataset(self):
        t = np.linspace(self.time_start, self.time_stop, self.length)
        mean = self.mean.get_value().flatten()
        std = self.variance.get_value().flatten()
        amp = self.amplitude.get_value().flatten()
        freq = self.frequency.get_value().flatten()
        y = mean + np.random.normal(0, std**2, self.length)
        y += amp * np.cos(freq * 2 * np.pi * t)

        if not isinstance(self.mean, SteppedChange):
            y[:self.location] += y[self.location:][0] - y[:self.location][-1]
        return t, y
