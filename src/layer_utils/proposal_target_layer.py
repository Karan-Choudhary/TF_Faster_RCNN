import numpy as np
import numpy.random as npr
from model_utils.bbox_overlaps import bbox_overlaps
from model_utils.bbox_transform import bbox_transform


def read_params(config_path):
    with open(config_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal classification
    labels and bounding-box regression targets.
    Proposal RoIs (0,x1,y1,x2,y2) coming from RPN
    """

    config = read_params('params.yaml')
    NUM_IMAGES = config['proposal_target_layer']['num_images']
    BATCH_SIZE = config['proposal_target_layer']['batch_size']
    FG_FRACTION = config['proposal_target_layer']['fg_fraction']

    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    all_scores = np.vstack((all_scores, zeros))

    # num_images = 1
    rois_per_image = BATCH_SIZE / NUM_IMAGES
    fg_rois_per_image = np.round(FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes)

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    config = read_params('params.yaml')
    BBOX_INSIDE_WEIGHTS = config['proposal_target_layer']['bbox_inside_weights']

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """
    Compute bounding-box regression targets for an image.
    """
    config = read_params('params.yaml')
    BBOX_NORMALIZE_MEANS = config['proposal_target_layer']['bbox_normalize_means']
    BBOX_NORMALIZE_STDS = config['proposal_target_layer']['bbox_normalize_stds']

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    targets = ((targets - np.array(config.TRAIN.BBOX_MEANS)) /
               np.array(config.TRAIN.BBOX_STDS))
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """
    Generate a random sample of RoIs comprising foreground and background examples.
    """
    # overlaps: (rois x gt_boxes)

    config = read_params('params.yaml')
    FG_THRESH = config['proposal_target_layer']['fg_thresh']
    BG_THRESH_HI = config['proposal_target_layer']['bg_thresh_hi']
    BG_THRESH_LO = config['proposal_target_layer']['bg_thresh_lo']

    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]

    # Ensure that a fixed number of regions are sampled
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(
            fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(
            bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(
            rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(
            rois_per_image), replace=to_replace)
        fg_rois_per_image = 0

    # the indices that we are selection (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
        bbox_target_data, num_classes)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
