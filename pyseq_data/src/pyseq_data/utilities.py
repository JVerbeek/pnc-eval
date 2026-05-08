import numpy as np 
import scipy.stats as ss
import yaml 

def save_data_to_npz(Xs, ys, fn="test_change_type"):
    np.savez(fn, Xs=Xs, ys=ys)
    return fn

def get_distribution(dname: str, params):
    """Convert distribution name to scipy.stats object"""
    return getattr(ss, dname)(*params)

def read_config_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)