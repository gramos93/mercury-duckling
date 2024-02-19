from functools import partial

from typing import Dict, Any
import torch.nn as nn
from torch import hub, cat
from torch.nn.functional import interpolate


class DinoV2(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_feats_sizes = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.config = config
        if self.config["backbone_size"] not in backbone_archs.keys():
            raise ValueError(
                f"Expected `backbone_size` to be one of {backbone_archs.keys()}"
            )
        backbone_arch = backbone_archs[self.config["backbone_size"]]
        backbone_name = f"dinov2_{backbone_arch}"
        self.backbone = hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=self.config["out_indices"],  # [8, 9, 10, 11]
            reshape=True,
        )
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        decode_head_width = (
            len(self.config["out_indices"])
            * backbone_feats_sizes[self.config["backbone_size"]]
        )
        self.decode_head = nn.Sequential(
            nn.BatchNorm2d(decode_head_width),
            nn.Conv2d(
                decode_head_width,
                self.config["classes"],
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
        )

    def forward(self, x: Any):
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = cat(x, dim=1)
        x = self.decode_head(x)
        return interpolate(x, (H, W), mode="bilinear", align_corners=True)
