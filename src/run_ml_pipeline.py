"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData


class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        # Point this to your clean Section 1 dataset (images/ + labels/)
        self.root_dir = "/home/workspace/section1/out"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/home/workspace/section2/out"


if __name__ == "__main__":
    c = Config()

    print("Loading data...")
    data = LoadHippocampusData(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)

    # 60 / 20 / 20 split
    keys = np.arange(len(data))
    np.random.seed(42)
    np.random.shuffle(keys)

    n = len(keys)
    split = {
        "train": keys[:int(0.6 * n)],
        "val": keys[int(0.6 * n):int(0.8 * n)],
        "test": keys[int(0.8 * n):]
    }

    exp = UNetExperiment(c, split, data)
    exp.run()

    results_json = exp.run_test()
    results_json["config"] = vars(c)

    os.makedirs(exp.out_dir, exist_ok=True)
    with open(os.path.join(exp.out_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print("Training and testing finished. Results written to", exp.out_dir)
