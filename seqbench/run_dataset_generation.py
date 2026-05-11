import json
import os
import numpy as np
import hashlib
import yaml

import pyseq
import pyseq_data.utilities as psdu
import pyseq_data.make_signal as psdf


def get_generator_kwargs(fname, kwargs_only=False):
    with open(fname) as f:
        hyperparams = yaml.safe_load(f)
        changepoint = hyperparams["changepoint"]
        properties = hyperparams["properties"]
        data_config = hyperparams["data_config"]
    if kwargs_only:
        return hyperparams
    return changepoint, properties, data_config, hyperparams

def get_generator_object_from_config(config="pyseq_data/src/pyseq_data/example_config.yaml"):
    changepoint, properties, data_config, hyperparams = get_generator_kwargs(config)
    kwargs = {}
    
    n_datapoints = data_config["n_datapoints"]
    kwargs["length"] = n_datapoints
    kwargs["time_start"], kwargs["time_stop"] = data_config["time_range"][0], data_config["time_range"][1]


    for name, params in properties.items():
        dist = psdu.get_distribution(params["dist"], params["dist_params"])
        sample = dist.rvs(1)
        value = psdf.Property(sample, dist)
        if params["change"]:
            after_change_value = dist.rvs(1)
            change_dict = { "location": changepoint["location"],
                            "before_change": value,
                            "after_change": after_change_value,
                            "length": n_datapoints
                            }
            if changepoint["stepped"]:
                result = psdf.SteppedChange(**change_dict)
            else:
                result = psdf.PolyChange(**change_dict, order=changepoint["order"])
        else:
            result = value 
        kwargs[name] = result
            
    if data_config["oscillating"]:
        ds = psdf.OscillationData(**kwargs, location=changepoint["location"])
    else: 
        kwargs.pop("frequency")
        kwargs.pop("amplitude")
        ds = psdf.ConstantData(**kwargs, location=changepoint["location"]) 
    return ds


def make_dataset(generator_hyperparameters, generator_name, set_name="train"):
    # Check if data has been generated before:

    # Convert hyperparameter kwargs dict to a folder name
    if generator_hyperparameters:
        # Create a concise, readable string from generator_kwargs for folder naming
        # Use a hash of the kwargs string for a robust folder
        generator_kwargs = get_generator_kwargs(generator_hyperparameters)
        generator_kwargs_str = json.dumps(generator_kwargs, sort_keys=True)
        hash_object = hashlib.sha256(generator_kwargs_str.encode())
        generator_kwargs_str = hash_object.hexdigest()[:32] #technically not unique, but hopefully fine

    else: # no hyperparameters provided, eenerator_kwargs.g. args.generator_hyperparameters is None
        generator_kwargs_str = "default"

    generated_data_folder = os.path.join("generated_data", generator_name.replace('.', '_'), generator_kwargs_str)

    os.makedirs(generated_data_folder, exist_ok=True)

    t_file = os.path.join(generated_data_folder, f"t_{set_name}.npz")
    y_file = os.path.join(generated_data_folder, f"y_{set_name}.npz")
    cps_file = os.path.join(generated_data_folder, f"cps_{set_name}.npz")
    params_file = os.path.join(generated_data_folder, f"params_{set_name}.json")

    if not os.path.exists(t_file) or not os.path.exists(y_file) or not os.path.exists(cps_file) or not os.path.exists(params_file):
        generator = get_generator_object_from_config(generator_hyperparameters)
        
        t, y, cps = generator.get_data(2)        # note:X_train is a N_d long list of matrices, Y_train is a N_d long list of indices of singular change points
        # Save the generated data with explicit keys
        np.savez_compressed(t_file, t=t) #check if this works fo    def get_data(n_datasets=1):
        np.savez_compressed(y_file, y=y)
        np.savez_compressed(cps_file, cps=cps)
        # Save params as JSON (assume params is a plain dict)
        #print(params)
        #with open(params_file, "w") as f:
        #    json.dump(params, f, indent=2)
    else:
    # Load the generated data using explicit keys
        t = np.load(t_file)["t"]
        y = np.load(y_file)["y"]
        cps = np.load(cps_file)["cps"]
        #with open(params_file, "r") as f:
        #    params = json.load(f)

    # Currently always standardize the y data, could implement generic preprocessing later?
    y = [(y_instance - y_instance.mean())/y_instance.std() for y_instance in y]

    return t, y, cps#, params


if __name__ == "__main__":
    print(get_generator_object_from_config().mean.get_value())