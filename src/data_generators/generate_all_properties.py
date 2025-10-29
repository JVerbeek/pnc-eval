import yaml
import numpy as np 

time_start = 0
time_end = 1
n_datapoints = 500
n_datasets = 10
glob_noise_params = [0, 1]
glob_noise_dist = "norm"

for c in ["noise", "perturbation", "frequency", "amplitude"]:
    for s in [True, False]:
        for osc in [True, False]:
            params = {
                "change_type": c, 
                "stepped": s, 
                "oscillating": osc, 
                "location": int(np.random.uniform(n_datapoints * 0.25, n_datapoints * 0.75)),
                "time_range": [time_start, time_end], 
                "n_datapoints": n_datapoints, 
                "glob_noise_dist": glob_noise_dist, 
                "glob_noise_params": glob_noise_params,
                "n_datasets": n_datasets
            }
            with open("config/change_types.yaml") as f:
                change_types = yaml.safe_load(f)
    
            params.update(change_types)
            s_str = "stepped" if s else "gradual"
            osc_str = "osc" if osc else "cons"

            with open(f"config/{c}-{s_str}-{osc_str}.yaml", "w") as f:
                yaml.dump(params, f)


