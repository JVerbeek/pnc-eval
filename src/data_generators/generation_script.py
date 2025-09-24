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

def generate_data_with_params(**params):
    """Creates a signal with a change point. For the given parameters, creates functions with those parameters, and compiles a signal additively."""
    ts, xs = [], []
    stepped = params["stepped"]
    location = params["location"]
    print(location)
    t = np.linspace(*params["time_range"], params["n_datapoints"])
    t_before, t_after = t[:location], t[location:]
    location = t[location]
    for t in (t_before, t_after):  # for both pre- and post-change, generate appropriate functions
        mean_fun = mean_fn(**params["params_perturbation"], location=location, stepped=stepped)
        noise_fun = noise_fn(**params["params_noise"], location=location, stepped=stepped)
        x = generate_constant_signal(0, 0.1, len(t))
        if params["oscillating"]:  # if it is an oscillating signal, also generate & use frequencx and amplitude functions
            freq_fun = freq_fn(**params["params_freq"], location=location, stepped=stepped)
            amp_fn = amplitude_fn(**params["params_amp"], location=location, stepped=stepped)
            x += amp_fn(t) * np.cos(freq_fun(t) * t) + mean_fun(t) + noise_fun(t)
        else:   # else, it is a "regular" signal. 
            x += noise_fun(t) + mean_fun(t)
        xs.append(x)
        ts.append(t)
    x0_bias = xs[0] - xs[0][-1]
    x1_bias = xs[1] - xs[1][2]
    t = np.concatenate(ts, 0)
    x = np.concatenate((x0_bias, x1_bias), 0)
    return t, x, location

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
    return (np.random.uniform(1, 50*time_range[1]) / time_range[1]) * (np.pi)

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

def sample_random_global_properties(n_datapoints):
    """Sample properties of the change and signal.

    Arguments:
        n_datapoints -- number of datapoints in the signal.

    Returns:
        dict:
            stepped: whether the change should be abrupt or gradual
            oscillating: whether the signal should oscillate
            location: location of the change
            change type: whether the change is in perturbation, noise, amplitude or frequency
    """
    stepped = np.random.randint(0, 2, 1).astype(bool).tolist()[0]   # Sample whether change is abrupt or gradual.
    oscillating = np.random.randint(0, 2, 1).astype(bool).tolist()[0]   # Sample whether data is oscillating.
    location = np.random.randint(0, n_datapoints, 1)[0]   # Sample random location in range.

    change_type = np.eye(4)[np.random.choice(4, 1)][0]   # Randomize change types one-hot.

    if not oscillating and np.any(change_type[2:]):  # Reroll if the sampled change type is oscillatory while the signal is not
        change_type[0:2] = np.eye(2)[np.random.choice(2, 1)] 
    return dict(zip(("stepped", "oscillating", "location", "change_type", "n_datapoints"), (stepped, oscillating, location, change_type, n_datapoints)))

def sample_parameters(time_range, change_type):
    """Construct dictionary of parameters for different change types. 
    For each of the possible change types, we sample a value. If the change should occur, then we sample a different value
    for the second part of the signal. If the change should not occur, then that value will be the same in the second part of the signal. 

    Arguments:
        time_range -- tuple: range of time (e.g. start time is 0, end time is 1)
        change_type -- boolean array: one-hot vector length 4 determining which change to generate

    Returns:
        param_dict:
            params_perturbation (b, nu): perturbation before and after the change. b is the bias towards which the shift takes place, nu is a possible exponent of the shift, such that we can also generate polynomial trends.
            params_noise (sigma0, sigma1): noise variance before and after the change
            params_amp (a0, a1): amplitude before and after the change
            params_freq (f0, f1): frequency before and after the change
    """
    param_dict = {"time_range": time_range}   # Dict to store dicts in
    for i, f in enumerate([sample_perturbation, sample_noise, sample_amp, sample_freq]):  # Sample some values and one change
        fname = f.__name__.split("_")[-1]
        val0 = f(time_range)  # Sample a value
        if fname == "perturbation":
            val1 = 0
        else:
            val1 = val0 # Set second value to same value as first
        if change_type[i] == 1:  # If the value of the change type vector at that index is 1, that is the changing property
            if fname == "perturbation":
                val1 = sample_exponent(time_range)
            else:
                val1 = f(time_range)  # set a different second value  (that will be the change)
        keys = get_param_keys(fname)
        param_dict[f"params_{fname}"] = dict(zip(keys, (val0, val1)))
    return param_dict

def generate_parameters(n_datasets, n_datapoints, time_range, signal_properties):
    """Generate dictionary of parameters for each of the datasets we want to generate. This dictionary can be passed directly into the data generation function for each of the datasets. 
    The resulting dictionary structure is the union of the dictionary resulting from sample_global_properties and sample_parameters.
    """
    param_dict = {}
    for n in range(n_datasets):  # For each dataset we want to generate
        param_dict["dataset_"+str(n)] = sample_parameters(time_range, signal_properties["change_type"])
        param_dict["dataset_"+str(n)].update(signal_properties)
    return param_dict

def generate_datasets(datapoints=10000, datasets=5, time_range=(0, 1), change_type="perturbation"):
    param_dict = generate_parameters(datasets, datapoints, time_range)
    for (_, params) in param_dict.items():
        t, x, y = generate_data_with_params(**params)   # vector, matrix, index
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
