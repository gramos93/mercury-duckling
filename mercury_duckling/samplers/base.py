from typing import Tuple
import numpy as np
from torch import Tensor
from matplotlib import pyplot as plt
from skimage.measure import regionprops


class BaseSampler:
    def __init__(self, seed: int = 42, neg_boundary=0.125):
        super().__init__()
        np.random.seed(seed)
        self._boundary = neg_boundary
        self._num_prompts = 1

    def _reset(self):
        self.prompts = []
        self._mask = None
        self._continue = True

    def interact(self, mask, outs=None, type="point"):
        assert type in [
            "point",
            "bbox",
        ], "Invalid type. Must be one of 'point' or 'bbox'."

        self._reset()
        self._mask = mask.astype(bool)
        self._region = self.prepare_mask(mask)
        self._outs = outs
        self._type = type
        while self._continue:
            if self._type == "point":
                self.prompts.append(self._sample_points(self._region, self._outs))
            else:
                self.prompts.append(self._sample_bboxs(self._region, self._outs))
            self._check_stop()
            yield self.prompts

    def _check_stop(self):
        if len(self.prompts) == self._num_prompts:
            self._continue = False

    def _set_outputs(self, *args, **kwargs):
        pass

    def _sample_points(self, region, outs=None):
        return {
            "type": "point",
            # coordinates in col(x), row(y) format.
            "coords": [region.centroid[1], region.centroid[0]],
            "label": 1,
        }
    def _get_scaled_bbox(self, region):
        # bbox in (min_row, min_col, max_row, max_col) format.
        ymin, xmin, ymax, xmax = region.bbox
        # make the bbox bigger by a factor of self.boundary.
        height = ymax - ymin
        width = xmax - xmin
        ymin = max(0, ymin - self._boundary * height)
        xmin = max(0, xmin - self._boundary * width)
        ymax = min(self._mask.shape[0], ymax + self._boundary * height)
        xmax = min(self._mask.shape[1], xmax + self._boundary * width)
        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def _sample_bboxs(self, region, outs=None):
        return {
            "type": "bbox",
            "coords": self._get_scaled_bbox(region),  # Format for SAM is tl,br
            "label": 1,
        }

    def prepare_mask(self, mask):
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy()

        region = regionprops(mask)[0]
        return region

    # From https://github.com/facebookresearch/segment-anything
    def show_points(coords, labels, ax, marker_size=150):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="o",
            s=marker_size,
            edgecolor="black",
            linewidth=1.25,
            alpha=0.8,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="X",
            s=marker_size,
            edgecolor="black",
            linewidth=1.25,
            alpha=0.8,
        )

    # From https://github.com/facebookresearch/segment-anything
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
