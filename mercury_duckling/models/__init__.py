import torch.nn as nn
from .predictor import BasePredictor, SamPredictor, RITMPredictor
from .segmentors import DinoV2

segmentor_registry = {"DINOV2": DinoV2}
predictor_registry = {
    "SAM": SamPredictor,
    "RITM": RITMPredictor,
    "IUNET": False,
}


def build_predictor(cfg) -> BasePredictor:
    if predictor := predictor_registry.get(cfg.selected_model, False):
        return predictor(cfg.model)
    else:
        raise KeyError(f"Predictor {cfg.selected_model} not implemented.")


def build_segmentor(cfg) -> nn.Module:
    if segmentor := segmentor_registry.get(cfg.selected_model, False):
        return segmentor(cfg.model)
    else:
        raise KeyError(f"Segmentor {cfg.selected_model} not implemented.")
