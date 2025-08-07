"""Data generator"""
import numpy as np 
import scipy 
import yaml
 
def generate_mean_change(**params):
    """Generate data"""
    Xs, ys = [], []
    for _ in range(params["n_datasets"]):
        change_magnitude = np.random.randint(-10, 10)   # Uniform, for now
        X = np.linspace(0, 1, params["n_datapoints"])
        y = scipy.stats.norm.rvs(loc=0, scale=0.1, size=params["n_datapoints"])   # Gaussian, for now
        y[:params["i_changepoint"]] += change_magnitude
        _, _ = Xs.append(X), ys.append(y)
    return Xs, ys

def save_data_to_npz(Xs, ys, fn="test_change_type"):
    np.savez(fn, Xs=Xs, ys=ys)
    return fn

def test():
    params = {}
    with open("example.yaml", encoding="utf-8") as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(params)
    Xs, ys = generate_mean_change(**params)
    location = save_data_to_npz(Xs, ys)
    print(f"Saved data to {location}.npz")
    return

if __name__=="__main__":
    test()
    ind = np.random.randint(0, 100)
    data = np.load("test_change_type.npz")
    X_datasets, y_datasets = data["Xs"], data["ys"]
    
    import matplotlib.pyplot as plt 
    plt.plot(X_datasets[ind], y_datasets[ind])   # Note the row-based indexing
    plt.show()