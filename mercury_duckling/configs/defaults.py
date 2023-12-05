# This file contains the default configurations parameters for the project.

SAM_THERMAL = {
    "origin": __file__,
    "device": "cpu",
    "type": "thermal",
    "logging": {"tags": "SAM"},
    "sampler": {
        "type": "points",
        "method": "clicker",
        "args": {
            "click_limit": 20
        },
    },
    "dataset": {
        "root": "../data/merged_dataset_delam",
        "annFile": "delamination.json",
    },
    "model": {
        "checkpoint": "checkpoints/sam_vit_h_4b8939.pth",
        "type": "vit_h",
    },
}

RITM_THERMAL = {
    "origin": __file__,
    "device": "cpu",
    "type": "thermal",
    "logging": {"tags": "RITM"},
    "sampler": {
        "type": "points",
        "method": "clicker",
        "args": {
            "click_limit": 20
        },
    },
    "dataset": {
        "root": "../data/merged_dataset_delam",
        "annFile": "delamination.json",
    },
    "model": {
        "checkpoint": "checkpoints/coco_lvis_h18_itermask.pth",
        "threshold": 0.49,
    },
}

UNET_THERMAL = {
    "origin": __file__,
    "device": "cpu",
    "type": "thermal",
    "logging": {"tags": "InterUnet"},
    "sampler": {
       "type": "points",
        "method": "clicker",
        "args": {
            "click_limit": 20
        },
    },
    "dataset": {
        "root": "../data/merged_dataset_delam",
        "annFile": "delamination.json",
    },
    "model": {
        "checkpoint": "checkpoints/InterSegSynthFT.pth",
    },
}