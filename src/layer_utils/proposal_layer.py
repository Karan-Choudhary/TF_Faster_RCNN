import tensorflow as tf
import yaml
import argparse
from model_utils.bbox_transform import clip_boxes_tf, bbox_transform_inv_tf

def read_params(config_path):
    with open(config_path,'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
    config = read_params('params.yaml')
    PRE_NMS_TOPN = config['proposal_layer']['pre_nms_topN']
    POST_NMS_TOPN = config['proposal_layer']['post_nms_topN']
    NMS_THRESH = config['proposal_layer']['nms_thresh']

    scores = rpn_cls_prob[:,:,:,num_anchors:]
    scores = tf.reshape(scores, shape=(-1,))
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
    proposals = clip_boxes_tf(proposals, im_info[:2])

    indices = tf.image.non_max_suppression(proposals, scores, max_output_size=POST_NMS_TOPN, iou_threshold=NMS_THRESH)
    boxes = tf.gather(proposals, indices)
    boxes = tf.to_float(boxes)
    scores = tf.gather(scores, indices)
    scores = tf.reshape(scores, shape=(-1, 1))

    batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
    rois = tf.concat([batch_inds, boxes], axis=1)

    return rois, scores