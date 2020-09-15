import os
import csv

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

DETECTION_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3
HAND_THRESHOLD = 0.9
ANCHORS_PATH = os.path.join("models", "anchors.csv")
with open(ANCHORS_PATH, "r") as csv_f: anchors = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]
BBOX_COLOR, LINE_WIDTH = (0, 255, 0), 2
KEY_POINT_COLOR, KEY_POINT_WIDTH, KEY_POINT_RADIUS = (20, 20, 20), 3, 3
# KEY_POINT_COLOR, KEY_POINT_WIDTH, KEY_POINT_RADIUS = (255, 255, 0), 3, 3
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
TARGET_TRIANGLE = np.float32([[128, 128], [128, 0], [0, 128]])
TARGET_BOX = np.float32([[0, 0, 1], [256, 0, 1], [256, 256, 1], [0, 256, 1], ])
SKELETON_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
SKELETON = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
BOX_SHIFT = 0.2
BOX_ENLARGE = 1.3


def preprocess(bgr_image, w, h):
    # convert to rgb
    rgb_image = bgr_image[:, :, ::-1]
    # pad to square and resize
    shape = np.r_[rgb_image.shape]
    padding = (shape.max() - shape[:2]).astype('uint32') // 2
    rgb_image = np.pad(rgb_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
    padding_image = rgb_image.copy()

    rgb_image = cv2.resize(rgb_image, (w, h))
    # normalize
    input_image = np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))
    padding_image = np.ascontiguousarray(2 * ((padding_image / 255) - 0.5).astype('float32'))
    return input_image, padding_image, padding


def detect_palm(input_image, palm_detector, input_details, output_details):
    palm_detector.set_tensor(input_details[0]['index'], input_image.reshape(1, 256, 256, 3))
    palm_detector.invoke()
    output_reg = palm_detector.get_tensor(output_details[0]['index'])[0]
    output_clf = palm_detector.get_tensor(output_details[1]['index'])[0, :, 0]
    return output_reg, output_clf


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def fast_nms(boxes, probabilities=None, overlap_threshold=0.3):
    """
    Algorithm to filter bounding box proposals by removing the ones with a too low confidence score
    and with too much overlap.
    Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes: List of proposed bounding boxes
    :param probabilities: scores of proposed bounding boxes
    :param overlap_threshold:  the maximum overlap that is allowed
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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    # return only the bounding boxes that were picked
    return pick


def get_res_from_palm_detector(output_reg, output_clf):
    # normalize scores to range 0 to 1 using sigmoid
    scores = sigmoid(output_clf)
    # filter by threshold
    output_reg = output_reg[scores > DETECTION_THRESHOLD]
    output_clf = output_clf[scores > DETECTION_THRESHOLD]
    candidate_anchors = anchors[scores > DETECTION_THRESHOLD]
    if output_reg.shape[0] == 0: print("No hands found")
    # get actual coordinate by pre-defined anchor
    moved_output_reg = output_reg.copy()
    moved_output_reg[:, :2] = moved_output_reg[:, :2] + candidate_anchors[:, :2] * 256
    # NMS for bounding boxes
    box_ids = fast_nms(moved_output_reg[:, :4], output_clf, NMS_THRESHOLD)
    # convert the coordinates back to the scale in original image size
    box_list = moved_output_reg[box_ids, :4].astype('int')
    side_list = []
    key_point_list = moved_output_reg[box_ids, 4:].reshape(-1, 7, 2)
    center_wo_offst = candidate_anchors[box_ids, :2] * 256
    for i in range(len(key_point_list)):
        key_point_list[i] = key_point_list[i] + center_wo_offst[i]
        x, y, w, h = box_list[i]
        side_list.append(max(w, h) * BOX_ENLARGE)
    return key_point_list, side_list


def get_triangle(kp0, kp2, dist=1):
    """get a triangle used to calculate Affine transformation matrix"""
    dir_v = kp2 - kp0
    dir_v /= np.linalg.norm(dir_v)
    dir_v_r = dir_v @ np.r_[[[0, 1], [-1, 0]]].T
    return np.float32([kp2, kp2 + dir_v * dist, kp2 + dir_v_r * dist])


def get_hand(input_image, key_points, side):
    source = get_triangle(key_points[0], key_points[2], side)
    source -= (key_points[0] - key_points[2]) * BOX_SHIFT
    transform_mat = cv2.getAffineTransform(source * max(input_image.shape) / INPUT_WIDTH, TARGET_TRIANGLE)
    hand = cv2.warpAffine(input_image, transform_mat, (INPUT_WIDTH, INPUT_HEIGHT))
    return hand, source


def detect_landmark(hand, landmark_model, input_details, output_details):
    landmark_model.set_tensor(input_details[0]['index'], hand.reshape(1, 256, 256, 3))
    landmark_model.invoke()
    landmark = landmark_model.get_tensor(output_details[0]['index']).reshape(-1, 2)
    is_hand = landmark_model.get_tensor(output_details[1]['index']) > HAND_THRESHOLD
    return landmark, is_hand


def convert_landmark_back(joints, source, padding, image):
    # projecting keypoints back into original image coordinate space
    transform_mat = cv2.getAffineTransform(source * max(image.shape) / INPUT_WIDTH, TARGET_TRIANGLE)
    transform_mat = np.pad(transform_mat.T, ((0, 0), (0, 1)), constant_values=1, mode='constant').T
    transform_mat[2, :2] = 0
    transform_mat_inv = np.linalg.inv(transform_mat)
    landmark = (np.pad(joints, ((0, 0), (0, 1)), constant_values=1, mode='constant') @ transform_mat_inv.T)[:, :2]
    landmark -= padding[::-1]

    # projecting keypoints back into input image coordinate space
    landmark_input = landmark + padding[::-1]
    landmark_input = landmark_input * INPUT_WIDTH / max(image.shape)
    return landmark, landmark_input


def triangle_to_bbox(source):
    # plain old vector arithmetics
    bbox = np.c_[
        [source[2] - source[0] + source[1]],
        [source[1] + source[0] - source[2]],
        [3 * source[0] - source[1] - source[2]],
        [source[2] - source[1] + source[0]],
    ].reshape(-1, 2)
    return bbox


def get_res_from_prev_res(prev_res):
    new_key_point_list = []
    side_list = []
    for key_point in prev_res:
        new_key_point_list.append(key_point[[0, 1, 9, 2, 5, 13, 17]])
        side_list.append(BOX_ENLARGE * max_dist(key_point[[0, 1, 9, 2, 5, 13, 17]]))
    return new_key_point_list, side_list


def max_dist(points):
    d = pdist(points)
    d = squareform(d)
    return np.nanmax(d)


def draw_landmark(image, landmark):
    for j in range(5):
        for k in range(4):
            cv2.line(image,
                     tuple(landmark[SKELETON[j][k]].astype('int')),
                     tuple(landmark[SKELETON[j][k + 1]].astype('int')),
                     SKELETON_COLORS[j],
                     3)

    for point in landmark:
        cv2.circle(image, (int(point[0]), int(point[1])), KEY_POINT_RADIUS, KEY_POINT_COLOR, KEY_POINT_WIDTH)

    return image


def draw_landmark_as_palm(image, landmark, box):
    landmark_middle = landmark[9]
    box_center = box.sum(axis=0) / len(box)
    bbox_pred = box + (landmark_middle - box_center)
    cx, cy, w, h = bbox_pred
    x1, y1, x2, y2 = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, LINE_WIDTH)

    landmark = landmark[[0, 1, 9, 2, 5, 13, 17]].astype('int')
    for key_point in landmark:
        cv2.circle(image, (int(key_point[0]), int(key_point[1])), KEY_POINT_RADIUS, KEY_POINT_COLOR, KEY_POINT_WIDTH)
    return image
