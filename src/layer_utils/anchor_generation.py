from __future__ import absolute_import
import numpy as np


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