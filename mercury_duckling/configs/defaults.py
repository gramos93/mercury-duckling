# This file contains the default configurations parameters for the project.

DEFAULT_THERMAL = {
    "origin": __file__,
    "device": "cuda:2",
    "type": "thermal",
    "logging": {},
    "num_epochs": 105,
    "val_interval": 4,
    "dataset": {
        "root": "../data/merged_dataset_delam",
        "annFile": "delamination.json",
        "batch_size": 32,
        "data_split": 0.75,
    },
    "model": {
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "classes": 1,
    },
    "optimizer": {
        "lr": 0.0001,
    #    "momentum": 0.9,
    #    "weight_decay": 0.0001
    },
    "scheduler": {
        "step": {
            "milestones": [40],
            "gamma": 0.1
        }
    }
}

DEFAULT_CONCRETE = {
    "origin": __file__,
    "device": "cuda:0",
    "type": "concrete",
    "logging": {},
    "num_epochs": 105,
    "val_interval": 4,
    "dataset": {
        "root": "../data/merged_dataset_concrete",
        "annFile": "concrete.json",
        "batch_size": 32,
        "data_split": 0.75,
    },
    "model": {
        "in_channels": 3,
        "classes": 1,
    },
    "optimizer": {
        "lr": 0.0001,
    },
    "scheduler": {
        # "cosine": {
        #     "T_0": 10,
        #     "T_mult": 2,
        # }
    }
}
