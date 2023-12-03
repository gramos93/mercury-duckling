from .base import BaseSampler
from .points import RandomPointSampler

__all__ = ["sampler_register", "BaseSampler"]

sampler_register = {
    "points":{
        "random": RandomPointSampler,
    },
    "bbox": {
        "base": BaseSampler,
    }
}