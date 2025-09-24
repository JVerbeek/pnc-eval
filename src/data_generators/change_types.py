CHANGE_PARAMS = {"perturbation": {"dist": "norm",
                                  "dist_params": (0, 1)
                                  },
                 "noise": {"dist": "uniform",
                           "dist_params": (0, 100)},
                 "amplitude": {"dist": "uniform",
                               "dist_params": (10, 100)},
                 "frequency": {"dist": "uniform",
                               "dist_params": (10, 100)},
                 }


PROPERTIES = {"change_type": "perturbation", "stepped": True, "oscillating": True, "location": 250, "time_range": (0, 1), "n_datapoints": 500}
# Yes, I know this is ugly and I am probably better of just writing yamls like God intended, but I really want to be able to test this quickly and not having to deal with file I/O is kind of nice.
