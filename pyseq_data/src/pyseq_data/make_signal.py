from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt  

@dataclass 
class Property:
    def __init__(self, value):
        self.value = value

    def asarray(self, length):
        return np.array([self.value] * length)

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

    def update(self, properties: dict):
        for fname, ftype in self.__annotations__.items():
            if fname in properties.keys():
                f = properties[fname]
                if isinstance(f, Property) or isinstance(f, ChangeDecorator):
                    setattr(self, fname, properties[fname])

@dataclass
class ChangeDecorator():
    def __init__(self, location=0, value=Property(0), after_change=1, length=100):
        self.location = location 
        self.value = value.value
        self.after_change = after_change
        self.length = length

    def __post_init__(self):
        for fname, ftype in self.__annotations__.items():
            f = getattr(self, fname)
            if isinstance(f, Property):
                f.value = np.array([f.value] * self.length)

    def get_value(self):
        f = self.get_fn() 
        return f(np.ones(len(self.value))) * value
        
@dataclass
class PolyChange(ChangeDecorator):
    def __init__(self, order=1, location=0, before_change=Property(0), after_change=1, length=100):
        super().__init__(location, before_change, after_change, length)
        self.order = order

    def get_fn(self):
        return lambda t: np.where(t < self.location, self.after_change, (self.after_change + t ** self.exponent *(self.before_change-self.after_change)) / (max(t) - min(t)))

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

    def get_data(self):
        t = np.linspace(self.time_start, self.time_stop, self.length)
        y = np.random.normal(self.mean.value, self.variance.value, self.length)
        return y 

@dataclass
class OscillationData(DataType):
    mean: Property
    variance: Property
    amplitude: Property
    frequency: Property

    def get_data(self):
        t = np.linspace(self.time_start, self.time_stop, self.length)
        y = np.random.normal(self.mean.value, self.variance.value, self.length)
        y += self.amplitude.value * np.cos(self.frequency.value * 2 * np.pi * t)
        return y

@dataclass
class ComplicatedCompositeFunction(DataType):
    mean: Property
    variance: Property
    carrier_frequency: Property
    modulating_frequency: Property

    def get_data(self):
        ## Define some complicated function of t here
        pass

# osc = OscillationData(
#     mean=Property(5), 
#     variance=Property(1), 
#     amplitude=PolyChange(
#         order=1, 
#         location=50, 
#         value=Property(5), 
#         after_change=3,
#         length=300), 
#     frequency=Property(5), 
#     time_start=0, 
#     time_stop=1,
#     length=300) 

