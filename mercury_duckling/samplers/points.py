from typing import Tuple
import numpy as np

from .base import BaseSampler


class RandomPointSampler(BaseSampler):
    def __init__(
        self,
        point_count: Tuple[int] = (2, 3),
        shuffle: bool = True,
        inverse: bool = True, # starts from positive points by default.
        neg_boundary: float = 0.25,
        seed: int = 42,
    ):
        # TODO: Add support for boundary scale for negative points.
        super().__init__(seed, neg_boundary)
        self._point_count = point_count
        self._inverse = inverse
        self._shuffle = shuffle

    def _reset(self):
        super()._reset()
        self.point_choices = []
        for count, type in zip(self._point_count, [1, 0]):
            self.point_choices.extend(count * [type])
        if self._inverse:
            self.point_choices = self.point_choices[::-1]
        if self._shuffle:
            np.random.shuffle(self.point_choices)

    def _check_stop(self):
        self._continue = len(self.point_choices) > 0

    def _sample_points(self, region, outs):
        point = self.point_choices.pop()
        if point:
            idx = np.random.randint(len(region.coords))
            coords = region.coords[idx][::-1].tolist()
        else:
            # xmin, ymin, xmax, ymax of the negative region close 
            # to the positive region.
            bbox = self._get_scaled_bbox(region)
            neg_y, neg_x = np.argwhere(np.logical_not(
                    self._mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            )).T + np.array([[bbox[1]], [bbox[0]]])
            idx = np.random.randint(len(neg_x))
            coords = [neg_x[idx], neg_y[idx]]
        
        return{
            "type": "point",
            "coords": coords,
            "label": point,
        }

    def _sample_bboxs(self, *args, **kwargs):
        raise NotImplementedError("This sampler does not sample bboxes.")
