from .base import BaseSampler
from .points import RandomPointSampler, ClickerSampler

__all__ = ["sampler_register", "BaseSampler"]

sampler_register = {
    "points":{
        "random": RandomPointSampler,
        "clicker": ClickerSampler,
    },
    "bbox": {
        "base": BaseSampler,
    }
}