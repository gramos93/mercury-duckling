from functools import partial
from typing import Any, List
import torch.nn as nn
from torch import hub, cat, ones_like
from torch.nn.functional import interpolate


class DinoV2(nn.Module):
    def __init__(
            self,
            size: str = "large",
            classes: int = 1,
            out_indices: List[int] = [8, 9, 10, 11],  
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
        self.colormap_head = nn.Sequential(
            nn.LayerNorm((1, 266, 322), eps=1e-06, elementwise_affine=True),
            nn.Conv2d(
                1,
                3,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.GELU(),
            nn.LayerNorm((3, 266, 322), eps=1e-06, elementwise_affine=True),
            nn.Conv2d(
                3,
                3,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.GELU(),
            nn.LayerNorm((3, 266, 322), eps=1e-06, elementwise_affine=True),
            nn.Conv2d(
                3,
                3,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.GELU(),
        )
        for param in self.colormap_head.parameters():
            if isinstance(param, nn.Conv2d):
                param.weight.data.copy_(
                    ones_like(param.weight.data, requires_grad=True, device=self.device)
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
        x = self.colormap_head(x)
        x = self.backbone(x)
        x = cat(x, dim=1)
        x = self.decode_head(x)
        return interpolate(x, (H, W), mode="bilinear", align_corners=True)
