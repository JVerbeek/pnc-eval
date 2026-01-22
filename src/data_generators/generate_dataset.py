import json
import os
import numpy as np

def generate_dataset(generator_kwargs, generator_fn, generator_hyperparameters, generator_name, set_name="train"):
    # Check if data has been generated before:

    # Convert hyperparameter kwargs dict to a folder name
    if generator_hyperparameters:
        # Create a concise, readable string from generator_kwargs for folder naming
        import hashlib
        # Use a hash of the kwargs string for a robust folder 
        generator_kwargs_str = json.dumps(generator_kwargs, sort_keys=True)
        
        hash_object = hashlib.sha256(generator_kwargs_str.encode())
        generator_kwargs_str = hash_object.hexdigest()[:32] #technically not unique, but hopefully fine

    else: # no hyperparameters provided, e.g. args.generator_hyperparameters is None
        generator_kwargs_str = "default"

    generated_data_folder = os.path.join("generated_data", generator_name.replace('.', '_'), generator_kwargs_str)

    os.makedirs(generated_data_folder, exist_ok=True)

    t_file = os.path.join(generated_data_folder, f"t_{set_name}.npz")
    y_file = os.path.join(generated_data_folder, f"y_{set_name}.npz")
    cps_file = os.path.join(generated_data_folder, f"cps_{set_name}.npz")
    params_file = os.path.join(generated_data_folder, f"params_{set_name}.json")


    if not os.path.exists(t_file) or not os.path.exists(y_file) or not os.path.exists(cps_file) or not os.path.exists(params_file):
        t, y, cps, params = generator_fn(properties=generator_kwargs)        # note:X_train is a N_d long list of matrices, Y_train is a N_d long list of indices of singular change points
        # Save the generated data with explicit keys
        np.savez_compressed(t_file, t=t) #check if this works for lists
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
