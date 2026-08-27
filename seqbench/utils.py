import importlib
import yaml
import sys
import numpy as np

def import_object_from_string(dotted_path):
    """Import an object (function, class, etc.) from a dotted module path like 'module.submodule.func'."""
    module_path, obj_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def handle_open_file(filename):
    try:
        if filename is None:
            return {}
        with open(filename, "r") as f:
                kwargs = yaml.safe_load(f)
        return {} if kwargs is None else kwargs
    except FileNotFoundError:
        print(f"File not found for {filename}. Did you supply the correct path?", file=sys.stderr)
        sys.exit()

def write_results(dirname, results):
    """ Writes results to file with name matching dataset hex name.
    results (list): list of results to write away. Minimally contains
    predictions, but can contain scores and regression predictions. 
    
    """
    ### Construct results_dict adaptively
    names = ["predictions", "scores", "regression_predictions"]
    results_dict = {names[i]: r for i, r in enumerate(results)}
    # Save results (unpack results_dict to kwargs, so that we have named arrays)
    np.savez(dirname+"/results.npz", **results_dict)
    import os
    print("Wrote results to", dirname, os.path.isfile(dirname+"results.npz"))
    return
