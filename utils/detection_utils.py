import cv2
import tensorflow as tf
import numpy as np

import csv
import os

DETECTION_THRESHOLD = 0.9
NMS_THRESHOLD = 0.5
ANCHORS_PATH = os.path.join("models", "anchors.csv")
with open(ANCHORS_PATH, "r") as csv_f: anchors = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]
BBOX_COLOR, LINE_WIDTH = (0, 255, 0), 2
KEY_POINT_COLOR, KEY_POINT_WIDTH, KEY_POINT_RADIUS = (255, 255, 0), 3, 3
INPUT_WIDTH = 256
INPUT_HEIGHT = 256


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def non_max_suppression_fast(boxes, probabilities=None, overlap_threshold=0.3):
    """
    Algorithm to filter bounding box proposals by removing the ones with a too low confidence score
    and with too much overlap.
    Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes: List of proposed bounding boxes
    :param overlap_threshold: the maximum overlap that is allowed
    :return: filtered boxes
    """
    # if there are no boxes, return an empty list
    if boxes.shape[1] == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] - (boxes[:, 2] / [2])  # center x - width/2
    y1 = boxes[:, 1] - (boxes[:, 3] / [2])  # center y - height/2
    x2 = boxes[:, 0] + (boxes[:, 2] / [2])  # center x + width/2
    y2 = boxes[:, 1] + (boxes[:, 3] / [2])  # center y + height/2

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = boxes[:, 2] * boxes[:, 3]  # width * height
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probabilities is not None:
        idxs = probabilities

    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))
    # return only the bounding boxes that were picked
    return pick


def normalize_image(rgb_image):
    return np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))


def preprocess(bgr_image, w, h):
    # convert to rgb
    rgb_image = bgr_image[:, :, ::-1]
    # pad to square and resize
    shape = np.r_[rgb_image.shape]
    padding = (shape.max() - shape[:2]).astype('uint32') // 2
    rgb_image = np.pad(rgb_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
    padding_image = rgb_image.copy()

    rgb_image = cv2.resize(rgb_image, (w, h))
    rgb_image = np.ascontiguousarray(rgb_image)
    # normalize
    rgb_image = normalize_image(rgb_image)
    # reshape as input shape
    rgb_image = rgb_image[tf.newaxis, ...]
    return rgb_image, padding_image, padding


def extract_bboxes_and_keypoints(output_reg, output_clf, padding, original_image):
    size = max(original_image.shape)
    scores = sigmoid(output_clf)
    output_reg = output_reg[scores > DETECTION_THRESHOLD]
    output_clf = output_clf[scores > DETECTION_THRESHOLD]
    candidate_anchors = anchors[scores > DETECTION_THRESHOLD]
    # if output_reg.shape[0] == 0:
    #     print("No hands found")
    moved_output_reg = output_reg.copy()
    moved_output_reg[:, :2] = moved_output_reg[:, :2] + candidate_anchors[:, :2] * 256
    box_ids = non_max_suppression_fast(moved_output_reg[:, :4], output_clf)
    center_wo_offst = candidate_anchors[box_ids, :2] * 256
    bboxes = moved_output_reg[box_ids, :4].astype('int')
    keypoints_set = output_reg[box_ids, 4:].reshape(-1, 7, 2)
    for i in range(len(keypoints_set)):
        keypoints_set[i] = keypoints_set[i] + center_wo_offst[i]
    ori_bboxes = bboxes.copy()
    ori_keypoints_set = keypoints_set.copy()

    for i, bbox in enumerate(bboxes):
        cx, cy, w, h = bbox
        cy = cy * size // INPUT_HEIGHT
        cx = cx * size // INPUT_WIDTH
        h_resize = h * size // INPUT_HEIGHT
        w_resize = w * size // INPUT_WIDTH
        cx -= padding[1]
        cy -= padding[0]
        x1, y1, x2, y2 = (cx - w_resize // 2, cy - h_resize // 2, cx + w_resize // 2, cy + h_resize // 2)
        bboxes[i] = np.array((x1, y1, x2, y2))

    for i, keypoints in enumerate(keypoints_set):
        for j, keypoint in enumerate(keypoints):
            tmp_keypoint = keypoint.copy()
            tmp_keypoint[0] = tmp_keypoint[0] * size // INPUT_WIDTH
            tmp_keypoint[1] = tmp_keypoint[1] * size // INPUT_HEIGHT
            tmp_keypoint[0] -= padding[1]
            tmp_keypoint[1] -= padding[0]
            keypoints_set[i][j] = (int(tmp_keypoint[0]), int(tmp_keypoint[1]))

    return bboxes, keypoints_set, ori_bboxes, ori_keypoints_set


def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        image = draw_bbox(image, bbox)
    return image


def draw_bbox(image, bbox):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), BBOX_COLOR, LINE_WIDTH)
    return image


def draw_keypoints_set(image, keypoints_set):
    for keypoints in keypoints_set:
        image = draw_keypoints(image, keypoints)
    return image


def draw_keypoints(image, keypoints):
    for keypoint in keypoints:
        cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), KEY_POINT_RADIUS, KEY_POINT_COLOR, KEY_POINT_WIDTH)
    return image
