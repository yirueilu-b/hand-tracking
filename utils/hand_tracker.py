import cv2
from scipy.spatial.distance import pdist, squareform
import numpy as np
import tensorflow as tf

from hand_tracker_config import Config


class HandTracker:
    def __init__(self):
        self.config = Config()
        # load palm model
        self.palm_model = tf.lite.Interpreter(model_path=self.config.PALM_MODEL_PATH)
        self.palm_model.allocate_tensors()
        self.palm_input_details = self.palm_model.get_input_details()
        self.palm_output_details = self.palm_model.get_output_details()
        # load landmark model
        self.landmark_model = tf.lite.Interpreter(model_path=self.config.LANDMARK_MODEL_PATH)
        self.landmark_model.allocate_tensors()
        self.landmark_input_details = self.landmark_model.get_input_details()
        self.landmark_output_details = self.landmark_model.get_output_details()

        self.is_hand_missing = None
        self.previous_landmarks_list = []
        self.padding = 0

    @staticmethod
    def _normalize(rgb_image):
        return np.ascontiguousarray(2 * ((rgb_image / 255) - 0.5).astype('float32'))

    @staticmethod
    def _sigmoid(values):
        return 1 / (1 + np.exp(-values))

    @staticmethod
    def _fast_nms(boxes, probabilities=None, overlap_threshold=0.3):
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

    @staticmethod
    def _get_source_tri(kp0, kp2, dist=1):
        """get a triangle used to calculate Affine transformation matrix"""
        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)
        dir_v_r = dir_v @ np.r_[[[0, 1], [-1, 0]]].T
        return np.float32([kp2, kp2 + dir_v * dist, kp2 + dir_v_r * dist])

    @staticmethod
    def _max_dist(points):
        d = pdist(points)
        d = squareform(d)
        return np.nanmax(d)

    def preprocess(self, bgr_image):
        # convert to rgb
        rgb_image = bgr_image[:, :, ::-1]
        # pad to square
        shape = np.r_[rgb_image.shape]
        padding = (shape.max() - shape[:2]).astype('uint32') // 2
        self.padding = padding
        rgb_image = np.pad(rgb_image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
        pad_image = rgb_image.copy()
        # resize
        rgb_image = cv2.resize(rgb_image, (self.config.INPUT_WIDTH, self.config.INPUT_HEIGHT))
        # normalize
        input_image = self._normalize(rgb_image)
        pad_image = self._normalize(pad_image)
        return input_image, pad_image

    def detect_palm(self, input_image):
        self.palm_model.set_tensor(self.palm_input_details[0]['index'], input_image.reshape(1, 256, 256, 3))
        self.palm_model.invoke()
        output_reg = self.palm_model.get_tensor(self.palm_output_details[0]['index'])[0]
        output_clf = self.palm_model.get_tensor(self.palm_output_details[1]['index'])[0, :, 0]
        return output_reg, output_clf

    def get_res_from_palm_model(self, output_reg, output_clf):
        # normalize scores to range 0 to 1 using sigmoid
        scores = self._sigmoid(output_clf)
        # filter by threshold
        output_reg = output_reg[scores > self.config.DETECTION_THRESHOLD]
        output_clf = output_clf[scores > self.config.DETECTION_THRESHOLD]
        candidate_anchors = self.config.ANCHORS[scores > self.config.DETECTION_THRESHOLD]
        if output_reg.shape[0] == 0: print("No hands found")
        # get actual coordinate by pre-defined anchor
        moved_output_reg = output_reg.copy()
        moved_output_reg[:, :2] = moved_output_reg[:, :2] + candidate_anchors[:, :2] * 256
        # NMS for bounding boxes
        box_ids = self._fast_nms(moved_output_reg[:, :4], output_clf, self.config.NMS_THRESHOLD)
        # convert the coordinates back to the scale in original image size
        box_list = moved_output_reg[box_ids, :4].astype('int')
        side_list = []
        key_point_list = moved_output_reg[box_ids, 4:].reshape(-1, 7, 2)
        center_wo_offst = candidate_anchors[box_ids, :2] * 256
        for i in range(len(key_point_list)):
            key_point_list[i] = key_point_list[i] + center_wo_offst[i]
            x, y, w, h = box_list[i]
            side_list.append(max(w, h) * self.config.BOX_ENLARGE)
        return key_point_list, side_list

    def crop_hand(self, input_image, key_points, side):
        source = self._get_source_tri(key_points[0], key_points[2], side)
        source -= (key_points[0] - key_points[2]) * self.config.BOX_SHIFT
        transform_mat = cv2.getAffineTransform(source * max(input_image.shape) / self.config.INPUT_WIDTH,
                                               self.config.TARGET_TRIANGLE)
        hand = cv2.warpAffine(input_image, transform_mat, (self.config.INPUT_WIDTH, self.config.INPUT_HEIGHT))
        return hand, source

    def detect_landmark(self, hand):
        self.landmark_model.set_tensor(self.landmark_input_details[0]['index'], hand.reshape(1, 256, 256, 3))
        self.landmark_model.invoke()
        landmark = self.landmark_model.get_tensor(self.landmark_output_details[0]['index']).reshape(-1, 2)
        is_hand = self.landmark_model.get_tensor(self.landmark_output_details[1]['index']) > self.config.HAND_THRESHOLD
        return landmark, is_hand

    def project_landmarks(self, landmarks, source, image):
        # projecting landmarks back into original image coordinate space
        transform_mat = cv2.getAffineTransform(source * max(image.shape) / self.config.INPUT_WIDTH,
                                               self.config.TARGET_TRIANGLE)
        transform_mat = np.pad(transform_mat.T, ((0, 0), (0, 1)), constant_values=1, mode='constant').T
        transform_mat[2, :2] = 0
        transform_mat_inv = np.linalg.inv(transform_mat)
        landmarks = (np.pad(landmarks, ((0, 0), (0, 1)), constant_values=1, mode='constant') @ transform_mat_inv.T)[:,
                    :2]
        landmarks -= self.padding[::-1]

        # projecting landmarks back into input image coordinate space
        landmarks_input = landmarks + self.padding[::-1]
        landmarks_input = landmarks_input * self.config.INPUT_WIDTH / max(image.shape)
        return landmarks, landmarks_input

    def get_res_from_prev_res(self):
        key_point_list = []
        side_list = []
        for key_point in self.previous_landmarks_list:
            key_point_list.append(key_point[[0, 1, 9, 2, 5, 13, 17]])
            side_list.append(self.config.BOX_ENLARGE * self._max_dist(key_point[[0, 1, 9, 2, 5, 13, 17]]))
        return key_point_list, side_list

    def inference(self, bgr_image):
        self.is_hand_missing = len(self.previous_landmarks_list) < self.config.NUM_TRACK_HAND
        input_image, pad_image = self.preprocess(bgr_image)
        if self.is_hand_missing:
            print("Palm detector activated!")
            key_point_list, side_list = self.get_res_from_palm_model(*self.detect_palm(input_image))
        else:
            print("Palm detector not activated!")
            key_point_list, side_list = self.get_res_from_prev_res()
        self.previous_landmarks_list = []
        res_landmarks_list = []
        for i in range(len(key_point_list)):
            hand, source = self.crop_hand(pad_image, key_point_list[i], side_list[i])
            landmarks, is_hand = self.detect_landmark(hand)
            if is_hand:
                landmark, landmark_input = self.project_landmarks(landmarks, source, bgr_image)
                res_landmarks_list.append(landmark)
                self.previous_landmarks_list.append(landmark_input)
        return res_landmarks_list

    def draw_landmark(self, image, landmarks):
        for j in range(5):
            for k in range(4):
                cv2.line(image,
                         tuple(landmarks[self.config.SKELETON[j][k]].astype('int')),
                         tuple(landmarks[self.config.SKELETON[j][k + 1]].astype('int')),
                         self.config.SKELETON_COLORS[j],
                         3)

        for point in landmarks:
            cv2.circle(image, (int(point[0]), int(point[1])),
                       self.config.KEY_POINT_RADIUS, self.config.KEY_POINT_COLOR, self.config.KEY_POINT_WIDTH)

        return image
