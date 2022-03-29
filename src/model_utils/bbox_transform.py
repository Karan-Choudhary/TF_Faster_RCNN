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

def bbox_transform_inv_tf(boxes, deltas):
    """
    Compute bounding box regression targets for an image.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    
    boxes = tf.cast(boxes, deltas.dtype)
    original_width = boxes[:, 2] - boxes[:, 0] + 1.0
    original_height = boxes[:, 3] - boxes[:, 1] + 1.0
    original_ctr_x = boxes[:, 0] + 0.5 * original_width
    original_ctr_y = boxes[:, 1] + 0.5 * original_height

    target_dx = deltas[:, 0::4]
    target_dy = deltas[:, 1::4]
    target_dw = deltas[:, 2::4]
    target_dh = deltas[:, 3::4]

    pred_ctr_x = tf.add(tf.multiply(target_dx, original_width), original_ctr_x)
    pred_ctr_y = tf.add(tf.multiply(target_dy, original_height), original_ctr_y)
    pred_w = tf.multiply(tf.exp(target_dw), original_width)
    pred_h = tf.multiply(tf.exp(target_dh), original_height)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w*0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h*0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w*0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h*0.5)

    predicted_boxes = tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)
    return predicted_boxes

def clip_boxes_tf(boxes, im_info):
    """
    Clip boxes to image boundaries.
    boxes: [N, 4*num_classes]
    im_info: [image_height, image_width, scale_ratios]
    """
    # x1 >= 0
    boxes[:, 0::4] = tf.maximum(tf.minimum(boxes[:, 0::4], im_info[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = tf.maximum(tf.minimum(boxes[:, 1::4], im_info[0] - 1), 0)
    # x2 < im_info[1]
    boxes[:, 2::4] = tf.maximum(tf.minimum(boxes[:, 2::4], im_info[1] - 1), 0)
    # y2 < im_info[0]
    boxes[:, 3::4] = tf.maximum(tf.minimum(boxes[:, 3::4], im_info[0] - 1), 0)
    return boxes