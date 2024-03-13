import torch.nn as nn
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus, FPN
from .predictor import BasePredictor, SamPredictor, RITMPredictor, IUnetPredictor
from .segmentors import DinoV2

segmentor_registry = {
    "DINOV2": DinoV2,
    "Unet": Unet,
    "Unet++": UnetPlusPlus,
    "DeepLabV3+": DeepLabV3Plus,
    "FPN": FPN
}
predictor_registry = {
    "SAM": SamPredictor,
    "RITM": RITMPredictor,
    "IUNET": IUnetPredictor,
}


def build_predictor(cfg) -> BasePredictor:
    if predictor := predictor_registry.get(cfg.selected_model, False):
        return predictor(cfg.model)
    else:
        raise KeyError(f"Predictor {cfg.selected_model} not implemented.")


def build_segmentor(cfg) -> nn.Module:
    if segmentor := segmentor_registry.get(cfg.selected_model, False):
        return segmentor(**cfg.model.args)
    else:
        raise KeyError(f"Segmentor {cfg.selected_model} not implemented.")
