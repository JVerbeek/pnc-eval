import numpy as np 
import scipy.stats as ss
import yaml 

def save_data_to_npz(Xs, ys, fn="test_change_type"):
    np.savez(fn, Xs=Xs, ys=ys)
    return fn

def get_distribution(dname: str):
    """Convert distribution name to scipy.stats object"""
    return getattr(ss, dname)

def sample_from_distribution(distribution, distribution_params):
    sample = distribution.rvs(*distribution_params, 1)
    return sample

def read_config_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)