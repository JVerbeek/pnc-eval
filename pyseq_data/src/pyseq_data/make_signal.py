from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt  

@dataclass 
class Property:
    def __init__(self, value):
        self.value = value

    def asarray(self, length):
        arr =  np.array([self.value] * length)
        return arr

    def get_value(self):
        return self.value

@dataclass
class DataType:
    mean: Property
    variance: Property
    time_start: float
    time_stop: float 
    length: int

    def __post_init__(self):
        for fname, ftype in self.__annotations__.items():
            print(fname, ftype)
            f = getattr(self, fname)
            if isinstance(f, Property):
                f.value = f.asarray(self.length)

    def apply_change(self):
        for fname, ftype in self.__annotations__.items():
            if isinstance(ftype, ChangeDecorator):
                f = getattr(self, fname)
                f.value = f.get_value()

    def update(self, properties: dict):
        for fname, ftype in self.__annotations__.items():
            if fname in properties.keys():
                f = properties[fname]
                if isinstance(f, Property) or isinstance(f, ChangeDecorator):
                    setattr(self, fname, properties[fname])

    def get_data(self, n_datasets):
        ts, ys = np.zeros((n_datasets, self.length)), np.zeros((n_datasets, self.length))
        for i in range(n_datasets):
            t, y = self.get_single_dataset()
            ts[i] = t
            ys[i] = y
        return ts, ys, np.array([self.location] * n_datasets)

@dataclass
class ChangeDecorator():
    def __init__(self, location=0, value=Property(0), after_change=1, length=100):
        self.location = location 
        self.value = np.array([value.value] * length)
        self.after_change = after_change
        self.length = length

    def get_value(self):
        v = self.value
        return v
        
@dataclass
class PolyChange(ChangeDecorator):
    def __init__(self, order=1, location=0, before_change=Property(0), after_change=1, length=100):
        super().__init__(location, before_change, after_change, length)
        self.order = order

    def get_value(self):
        t = np.arange(self.length)
        t_prime = self.get_fn()(t)
        return t_prime

    def get_fn(self):
        return lambda t: np.where(t < self.location, self.value, self.value + ((t ** self.order) * np.abs(self.value - self.after_change)) / (max(t) - min(t)))

@dataclass
class SteppedChange(ChangeDecorator):
    def __init__(self, location=0, before_change=Property(0), after_change=1, length=100):
        super().__init__(location, before_change, after_change, length)

    def get_fn(self):
        return lambda t: np.where(t < self.location, self.before_change, self.after_change)

@dataclass
class ConstantData(DataType):
    mean: Property
    variance: Property
    location: 0

    def get_single_dataset(self):
        for prop in vars(self):
            if isinstance(prop, ChangeDecorator):
                self.location = prop.location
                setattr(self, prop, prop.get_value())
        t = np.linspace(self.time_start, self.time_stop, self.length)
        print(np.abs(self.variance.get_value()))
        print(self.mean.get_value().shape)
        y = np.random.normal(self.mean.get_value(), np.abs(self.variance.get_value())).flatten()
        return t, y 

@dataclass
class OscillationData(DataType):
    mean: Property
    variance: Property
    amplitude: Property
    frequency: Property

    def get_single_dataset(self):
        for prop in vars(self):
            if isinstance(prop, ChangeDecorator):
                setattr(self, prop, prop.get_value())
        t = np.linspace(self.time_start, self.time_stop, self.length)
        y = np.random.normal(self.mean.get_value(), np.abs(self.variance.get_value()), self.length)
        y += self.amplitude.get_value() * np.cos(self.frequency.get_value() * 2 * np.pi * t)
        return y
