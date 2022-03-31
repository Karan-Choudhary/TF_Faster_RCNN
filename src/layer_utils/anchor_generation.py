from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

def generate_anchors_pre_tf(height, width, feat_strides=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """
    A wrapper function to generate anchors given different scales and
    ratios.
    Create uniformly spaced grid with spacing equal to stride
    """
    shift_x = tf.range(width) * feat_strides # [0,16,32,48] width
    shift_y = tf.range(height) * feat_strides
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1]) # reshape to 1D
    shift_y = tf.reshape(shift_y, [-1]) 
    shifts = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1) # vertical stack by row
    K = tf.multiply(width,height)
    shifts = tf.transpose(shifts,shape=[1,K,4],perm=[1,0,2]) #reshaping into Kx1x4
    # basic 9 anchor boxes of shape (9,4)
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0] # number of anchors
    anchor_constants = tf.constant(anchors.reshape((1,A,4)), dtype=tf.float32) # reshape to 1x9x4
    length = K*A
    anchors_tf = tf.reshape(tf.add(anchor_constants,shifts),shape=[length,4]) # reshape to Kx9x4 and dd shift to anchors element wise
    return tf.cast(anchors_tf,tf.float32), length

def generate_anchors(base_size=16, ratios=[0.5,1,2], scales = 2**np.arange(3,6)):
    """
    Generate anchor windows by enumerating aspect ratio X
    scales wrt a reference (0,0,15,15) window.
    """
    base_anchor = np.array([1,1,base_size,base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors_list = []
    for i in range(ratio_anchors.shape[0]):
        anc = _scale_enum(ratio_anchors[i,:], scales)
        anchors_list.append(anc)
    anchors = np.vstack(anchors_list)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor
    """
    w, h, x_ctr, y_ctr = _extract_cor(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor
    """
    w, h, x_ctr, y_ctr = _extract_cor(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _make_anchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _extract_cor(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    # anchor is 1X4
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr