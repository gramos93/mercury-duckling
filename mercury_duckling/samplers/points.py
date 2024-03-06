from typing import Tuple
import numpy as np
from torch import Tensor
from cv2 import distanceTransform, DIST_L2

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
        """
        Sample a point from the region.

        Args:
            region (RegionProperties): The region to sample from. (see skimage.measure.regionprops)
            outs (_type_): The output of the model from the previous iteration.
        
        Returns:
            dict: A dictionary containing the sampled point.
                  Points coordinates are in (x, y) format. 
                  Labels are 1 for positive and 0 for negative.
                  Type is always "point".
        """
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
    
class ClickerSampler(BaseSampler):
    def __init__(
        self,
        click_limit: int = 20,
        seed: int = 42,
    ):
        super().__init__(seed)
        self._click_limit = click_limit

    def _reset(self):
        super()._reset()
        self.click_list = []
        self._gt_mask = None

    def interact(self, mask, outs=None):
        """
        Base function to interact with an interactive segmentation model.
        This function will yeild a prompt.

        Args:
            mask (np.array): ground thruth mask.
            outs (np.array, optional): Initial prediction mask. Defaults to None.

        Yields:
            list: List of promtps for the model to use. 
        """
        self._reset()
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy()

        self._gt_mask = self.prepare_mask(mask)
        self.not_clicked_map = np.ones_like(self._gt_mask, dtype=bool)
        self._outs = np.zeros_like(self._gt_mask) if outs is None else outs.astype(bool)
        while len(self.prompts) < self._click_limit:
            self.prompts.append(self._sample_points(self._outs, padding=False))
            yield self.prompts
    
    def set_outputs(self, outs):
        self._outs = self.prepare_mask(outs)
    
    def prepare_mask(self, mask):
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy()

        if mask.ndim > 2:
            single_dim = np.argwhere(np.array(mask.shape) == 1)
            if not single_dim.size == 1:
                raise ValueError("Input mask dims must de [1, H, W] or [H, W].")
            
            mask = mask.squeeze(single_dim.item())
        
        return mask.astype(bool)

    def _sample_points(self, pred_mask, padding=True):
        fn_mask = np.logical_and(self._gt_mask, np.logical_not(pred_mask))
        fp_mask = np.logical_and(np.logical_not(self._gt_mask), pred_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = distanceTransform(fn_mask.astype(np.uint8), DIST_L2, 0)
        fp_mask_dt = distanceTransform(fp_mask.astype(np.uint8), DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        self.not_clicked_map[coords_y[0], coords_x[0]] = False

        return {
            "type": "point",
            "coords": [coords_x[0], coords_y[0]],
            "label": is_positive,
        }
    