"""Data generator"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from change_types import CHANGE_PARAMS, PROPERTIES   # I don't like this, but it makes it much easier to understand what is going on
import scipy.stats as ss

def generate_constant_signal(loc, scale, n_datapoints):
    y = scipy.stats.norm.rvs(loc=loc, scale=scale, size=n_datapoints)   # Gaussian, for now
    return y

def linear_increase_fn(first_param, second_param,  location, stepped, exponent=1):
    """Generates some function with a change. If the change is a step, then the change jumps from first_param to second_param.
    If the change is not stepped, then the changes happens linearly and gradually, with rate-of-change (second_param - first_param).
    As an example, an stepped change in mean might have a jump from 0 to 10 at location 50, while a non-stepped change might start increasing with rate-of-change 10 at location 50.
    """
    if stepped:
        return lambda t: np.where(t < location, first_param, second_param)
    else:
        return lambda t: np.where(t < location, first_param, (first_param + (second_param - first_param) * t / (max(t) - min(t)) ** exponent))

def generate_data_with_params(**params):
    """Creates a signal with a change point. For the given parameters, creates functions with those parameters, and compiles a signal additively."""
    stepped = params["stepped"]
    location = int(params["location"])
    t = np.linspace(*params["time_range"], params["n_datapoints"])
    time_index = t[location]
    glob_noise_dist = get_distribution(params["glob_noise_dist"])
    glob_noise = sample_from_distribution(glob_noise_dist, params["glob_noise_params"])

    for p in params.keys():  # Update symbol table with appropriate function names
        pname = p.split("_")
        if pname[0] != "params":
            continue
        globals()[pname[1]+"_fun"] = linear_increase_fn(*params[p], location=time_index, stepped=stepped)

    if params["oscillating"]: # Function composition is something you have to be specific about
        x = amplitude_fun(t) * np.cos(frequency_fun(t) * t) + perturbation_fun(t) + noise_fun(t) * ss.norm.rvs(0, 0.1, params["n_datapoints"])
    else:
        x = noise_fun(t) + ss.norm.rvs(0, 0.1, params["n_datapoints"]) + perturbation_fun(t)

    if params["change_type"] == "perturbation": # (else you'd remove the step)
        x[:location] -= x[:location][-1] - x[location]
    elif params["change_type"] == "perturbation" and not params["stepped"]:  # then actually we do want the signals to attach
        x[:location] -= x[:location][-1] - x[location]

    return t, x, location

def save_data_to_npz(Xs, ys, fn="test_change_type"):
    np.savez(fn, Xs=Xs, ys=ys)
    return fn

def get_distribution(dname: str):
    """Convert distribution name to scipy.stats object"""
    return getattr(ss, dname)

def sample_from_distribution(distribution, distribution_params):
    sample = distribution.rvs(*distribution_params, 1)
    return sample


def sample_parameters(time_range, change_type):
    """Construct dictionary of parameters for different change types. 
    For each of the possible change types, we sample a value. If the change should occur, then we sample a different value
    for the second part of the signal. If the change should not occur, then that value will be the same in the second part of the signal.

    Arguments:
        time_range -- tuple: range of time (e.g. start time is 0, end time is 1)
        change_type -- boolean array: one-hot vector length 4 determining which change to generate

    Returns:
        param_dict -- dict: dictionary of the form {"property1": (before, after), "property2": (before, after), ... "propertyN": (before, after)}
    """
    param_dict = {"time_range": time_range}   # Dict to store dicts in
    for c in list(CHANGE_PARAMS.keys()):
        params = CHANGE_PARAMS[c]
        distribution_name = get_distribution(params["dist"])
        distribution_params = params["dist_params"]   # returns list of distribution parameters
        val0 = sample_from_distribution(distribution_name, distribution_params)
        val1 = sample_from_distribution(distribution_name, distribution_params)
        if c != change_type:   # If the change is not the one we are looking for, then the second value can be the same as the first (i.e. no change in that property)
             val1 = val0
        param_dict["params_"+c] = (val0, val1)
    print(param_dict)
    return param_dict

def generate_parameters(n_datasets, n_datapoints, time_range, properties):
    """Generate dictionary of parameters for each of the datasets we want to generate. This dictionary can be passed directly into the data generation function for each of the datasets.
    The resulting dictionary structure is the union of the dictionary resulting from sample_global_properties and sample_parameters.
    """
    param_dict = {}
    for n in range(n_datasets):  # For each dataset we want to generate
        param_dict["dataset_"+str(n)] = sample_parameters(time_range, properties["change_type"])
        param_dict["dataset_"+str(n)].update(properties)
    return param_dict

def generate_datasets(datapoints=10000, datasets=5, time_range=(0, 1)):
    param_dict = generate_parameters(datasets, datapoints, time_range, PROPERTIES)
    for (_, params) in param_dict.items():
        t, x, y = generate_data_with_params(**params)   # vector, matrix, index
        plt.plot(t[:int(params["location"])], x[:int(params["location"])], marker="x", color="black")
        plt.plot(t[int(params["location"]):], x[int(params["location"]):], marker="x",color="red")
        plt.xlabel("t")
        plt.xlabel("y")
        plt.show()

def test():
    generate_datasets(datapoints=10000, datasets=10, time_range=(0, 1))
    return

if __name__=="__main__":
    test()
