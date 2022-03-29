from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

def bbox_transform(original_rois, gt_rois):
    """
    Calculate the bounding box regression coefficients.
    """
    original_width = original_rois[:, 2] - original_rois[:, 0] + 1.0
    original_height = original_rois[:, 3] - original_rois[:, 1] + 1.0
    original_ctr_x = original_rois[:, 0] + 0.5 * original_width
    original_ctr_y = original_rois[:, 1] + 0.5 * original_height

    gt_width = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_height = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - original_ctr_x) / original_width
    target_dy = (gt_ctr_y - original_ctr_y) / original_height
    target_dw = np.log(gt_width / original_width)
    target_dh = np.log(gt_height / original_height)

    targets = np.vstack((target_dx, target_dy, target_dw, target_dh)).transpose()
    return targets