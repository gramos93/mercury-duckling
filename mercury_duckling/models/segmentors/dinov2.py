from functools import partial
from typing import Any, List
import torch.nn as nn
from torch import hub, cat
from torch.nn.functional import interpolate


class DinoV2(nn.Module):
    def __init__(
            self,
            size: str = "large",
            classes: int = 2,
            out_indices: List[int] = [8, 9, 10, 11],    
            **kwargs
        ) -> None:
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
        if size not in backbone_archs.keys():
            raise ValueError(
                f"Expected `backbone_size` to be one of {backbone_archs.keys()}"
            )
        backbone_arch = backbone_archs[size]
        backbone_name = f"dinov2_{backbone_arch}"
        self.backbone = hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )
        self.backbone.forward = partial(
            self.backbone.get_intermediate_layers,
            n=out_indices,
            reshape=True,
        )
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        decode_head_width = (
            len(out_indices)
            * backbone_feats_sizes[size]
        )
        self.decode_head = nn.Sequential(
            nn.BatchNorm2d(decode_head_width),
            nn.Conv2d(
                decode_head_width,
                classes,
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
