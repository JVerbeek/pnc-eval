import numpy as np
import matplotlib.pyplot as plt 
import scipy
import scipy.stats as ss
import yaml

from make_signal import Property, SteppedChange, PolyChange, ConstantData, OscillationData
import utilities as utils

def attach_signal_halves(x, location):
    if params["change_type"] != "perturbation": 
        x[:location] -= x[:location][-1] - x[location]

    elif params["change_type"] == "perturbation" and not params["stepped"]:  # then actually we do want the signals to attach
        x[:location] -= x[:location][-1] - x[location]
    return x

def parse_data_config(cfg: dict):
    if cfg["oscillating"]:
        data = OscillationData(mean=0, variance=0, frequency=0, amplitude=0, length=cfg["n_datapoints"], time_start=cfg["time_range"][0], time_stop=cfg["time_range"][1])
    else: 
        data = ConstantData(mean=0, variance=0, length=data_config["n_datapoints"], time_start=cfg["time_range"][0], time_stop=cfg["time_range"][1])
    return data

def get_changepoint(cfg: dict, before_change: Property, after_change):
    if cfg["stepped"]:
        return SteppedChange(location=cfg["location"], before_change=before_change, after_change=after_change)
    else:
        return PolyChange(location=cfg["location"], before_change=before_change, after_change=after_change)
    
def parse_properties(cfg: dict):
    property_dict = {}
    for k, v in cfg.items():
        if k == "changepoint":
            continue
        dist = utils.get_distribution(v["dist"])
        value = utils.sample_from_distribution(dist, v["dist_params"])
        if k == "variance":
            value = value ** 2
        property_dict[k] = Property(value[0])
        if v["change"]:  # If this property should have a change, set some variables. 
            change_type = k
            after_change = utils.sample_from_distribution(dist, v["dist_params"])
             
    # Change the property that should exhibit a change to a changepoint property.
    changepoint = get_changepoint(cfg["changepoint"], before_change=property_dict[change_type], after_change=after_change)
    property_dict[change_type] = changepoint
    return property_dict

def make_dataclass(config):
    dataclass = parse_data_config(config["data_config"])
    properties = parse_properties(config["properties"])
    dataclass.update(properties)
    time_range = config["data_config"]["time_range"]
    x = np.linspace(time_range[0], time_range[1], config["data_config"]["n_datapoints"])
    return x, dataclass

if __name__=="__main__":
    with open("/home/janneke/repos/pnc-eval/pyseq_data/src/pyseq_data/example_config.yaml") as f:
        dataclass_config = yaml.safe_load(f)
    ts, ys, cps = [], [], []
    for i in range(dataclass_config["data_config"]["n_datasets"]):
        t, dataclass = make_dataclass(dataclass_config)
        y = dataclass.get_data()
        ts.append(t), ys.append(y), cps.append(dataclass_config["properties"]["changepoint"]["location"])

    plt.plot(ts[0], ys[0])
    plt.axvline(ts[0][cps[0]])
    plt.show()