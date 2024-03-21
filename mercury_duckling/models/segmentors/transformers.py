from typing import Any, List
import torch.nn as nn
from torch.nn.functional import interpolate
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    PretrainedConfig
)


class ThermalSegFormer(nn.Module):
    def __init__(self, in_channels=3, classes=2, checkpoint=None) -> None:
        super(ThermalSegFormer, self).__init__()

        if checkpoint is not None:
            self._model_cfg = PretrainedConfig.from_pretrained(checkpoint)
            self._model_cfg.num_labels = classes
            self._model_cfg.num_channels = in_channels
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                config=self._model_cfg,
                ignore_mismatched_sizes=True,
            )
        else:
            self._model_cfg = SegformerConfig(
                num_channels=in_channels,
                num_labels=classes
            )
            self._model = SegformerForSemanticSegmentation(config=self._model_cfg)

        self._model.train()

    def forward(self, inpts: Any) -> Any:
        B, C, H, W = inpts.shape
        outs = self._model(pixel_values = inpts).logits
        return interpolate(
            outs,
            (H, W),
            mode="bilinear",
            # align_corners=True
        )