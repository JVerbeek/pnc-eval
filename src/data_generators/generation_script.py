"""Data generator"""
import matplotlib.pyplot as plt 
import numpy as np 
import scipy 
import yaml

def generate_constant_signal(loc, scale, n_datapoints):
    y = scipy.stats.norm.rvs(loc=loc, scale=scale, size=n_datapoints)   # Gaussian, for now
    return y 

def freq_fn(f0, f1, location=0, stepped=False):  # Create function with either a gradual or a stepped change in frequency
    if stepped:		    
	    return lambda t: np.where(t < location, f0, f1)   
    else:
        return lambda t: np.where(t < location, f0, ((f0 + (f1 - f0))*t) / (max(t) - min(t)))

def amplitude_fn(a0, a1, location=0, stepped=False):   # Create function with gradual or stepped change in frequency
    if stepped:
        return lambda t: np.where(t < location, a0, a1)
    else:
        return lambda t: np.where(t < location, a0, (a0 + (a1 - a0)) * t / (max(t)-min(t)))

def noise_fn(sigma0, sigma1, stepped=False, location=0, nu=1.0):   # Create function with gradual or stepped change in noise
    if stepped:
        return lambda t: np.where(t < location, sigma0, sigma1) 
    else:
        return lambda t: np.where(t < location, sigma0, ((sigma0 + (sigma1 - sigma0)) * t / (max(t)-min(t))) ** nu )
                                  
def mean_fn(b=1.0, nu=1.0, stepped=False, location=0):   # Create function with gradual or stepped change in mean (perturbation)
    if stepped:
        return lambda t: np.where(t < location, 0, b)
    else:
        return lambda t: (np.where(t < location, 0, (b * t) / (max(t)-min(t)))) ** nu
    
def generate_data_with_params(n_datapoints=100, params_freq={"f0": 100, "f1": 10}, params_amp={"a0" :10, "a1" :10}, params_noise={"sigma0":0.5, "sigma1":1.0, "nu":1}, params_perturbation={"b":0, "nu":1}, location=0, time_range=(0, 1), stepped=True, oscillating=True):
    ts, ys = [], []
    t = np.linspace(*time_range, n_datapoints)
    t_before, t_after = t[:location], t[location:]
    location = t[location]
    for t in (t_before, t_after):  # for both pre- and post-change, generate appropriate functions
        mean_fun = mean_fn(**params_perturbation, location=location, stepped=stepped)
        noise_fun = noise_fn(**params_noise, location=location, stepped=stepped)
        y = generate_constant_signal(0, 0.1, len(t))
        if oscillating:  # if it is an oscillating signal, also generate & use frequency and amplitude functions
            freq_fun = freq_fn(**params_freq, location=location, stepped=stepped)
            amp_fn = amplitude_fn(**params_amp, location=location, stepped=stepped)
            y += amp_fn(t) * np.cos(freq_fun(t) * t) + mean_fun(t) + noise_fun(t)
        else:   # else, it is a "regular" signal. 
            y += noise_fun(t) + mean_fun(t)
        ys.append(y)
        ts.append(t)
    y0_bias = ys[0] - ys[0][-1]
    y1_bias = ys[1] - ys[1][2]
    t = np.concatenate(ts, 0)
    y = np.concatenate((y0_bias, y1_bias), 0)
    return t, y

def save_data_to_npz(Xs, ys, fn="test_change_type"):
    np.savez(fn, Xs=Xs, ys=ys)
    return fn

def sample_perturbation(time_range): 
    return np.random.normal(0, 4)

def sample_exponent(time_range):
    return np.random.random() + 0.01

def sample_noise(time_range):
    return np.abs(np.random.normal(0, 4))

def sample_freq(time_range):
    return (np.random.uniform(1, time_range[1]) / time_range[1]) * (np.pi)

def sample_amp(time_range):
    return np.random.choice(np.arange(1, 20, 0.5))

def get_param_keys(fname):
    if fname == "perturbation":
        return ("b", "nu")
    if fname == "noise":
        return ("sigma0", "sigma1")
    if fname == "amp":
        return ("a0", "a1")
    if fname == "freq":
        return ("f0", "f1")
    return 

def generate_parameters(n_datasets, n_datapoints, time_range):
    stepped = np.random.randint(0, 2, n_datasets).astype(bool).tolist()   # Sample whether change is abrupt or gradual.
    oscillating = np.random.randint(0, 2, n_datasets).astype(bool).tolist()   # Sample whether data is oscillating.
    location = np.random.randint(0, n_datapoints, n_datasets)   # Sample random location in range.
    
    change_types = np.eye(4)[np.random.choice(4, n_datasets)]   # Randomize change types one-hot.
    
    for i in range(n_datasets): # If the dataset should not oscillate and we sampled amplitude or frequency changes, reroll:
        if not oscillating[i] and np.any(change_types[2:]):
            change_types[i][0:2] = np.eye(2)[np.random.choice(2, 1)]  # Assumes perturbation and noise are the first two columns of the change type matrix.
            
    param_dict = {}
    
    for n in range(n_datasets):
        param_dict["dataset_"+str(n)] = {}
        for i, f in enumerate([sample_perturbation, sample_noise, sample_amp, sample_freq]):
            fname = f.__name__.split("_")[-1]
            val0 = f(time_range)  # Sample a value
            if fname == "perturbation":
                val1 = 0
            else:
                val1 = val0 # Set second value to same value as first
            if change_types[n][i] == 1:  # If the value of the change type vector at that index is 1, that is the changing property
                if fname == "perturbation":
                    val1 = sample_exponent(time_range)
                else:
                    val1 = f(time_range)  # set a different second value  (that will be the change)
            keys = get_param_keys(fname)
            param_dict["dataset_"+str(n)]["params_"+fname] = dict(zip(keys, (val0, val1)))
    
        keys = ["stepped", "oscillating", "location", "n_datapoints", "time_range"]
        param_dict["dataset_"+str(n)].update(dict(zip(keys, (stepped[n], oscillating[n], location[n], n_datapoints, time_range)))) 
    return param_dict

def generate_datasets(datapoints=10000, datasets=5, time_range=(0, 1)):
    param_dict = generate_parameters(datasets, datapoints, time_range)
    for (_, params) in param_dict.items():
        t, y = generate_data_with_params(**params)
        plt.plot(t[:int(params["location"])], y[:int(params["location"])], marker="x", color="black")
        plt.plot(t[int(params["location"]):], y[int(params["location"]):], marker="x",color="red")
        plt.xlabel("t")
        plt.xlabel("y")
        plt.show()
    
def test():
    generate_datasets(datapoints=10000, datasets=10, time_range=(0, 1))
    return

if __name__=="__main__":
    test() 