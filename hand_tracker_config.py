import os
import csv
import numpy as np


class Config:
    def __init__(self):
        self.PALM_MODEL_PATH = os.path.join('models', 'palm_detection_without_custom_op.tflite')
        self.LANDMARK_MODEL_PATH = os.path.join('models', 'hand_landmark.tflite')
        anchors_path = os.path.join("models", "anchors.csv")
        with open(anchors_path, "r") as csv_f:
            self.ANCHORS = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]

        self.NUM_TRACK_HAND = 1
        self.DETECTION_THRESHOLD = 0.8
        self.NMS_THRESHOLD = 0.3
        self.HAND_THRESHOLD = 0.9
        self.BOX_SHIFT = 0.2
        self.BOX_ENLARGE = 1.3

        self.TARGET_TRIANGLE = np.float32([[128, 128], [128, 0], [0, 128]])
        self.TARGET_BOX = np.float32([[0, 0, 1], [256, 0, 1], [256, 256, 1], [0, 256, 1], ])

        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.INPUT_WIDTH = 256
        self.INPUT_HEIGHT = 256

        self.SKELETON_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
        self.SKELETON = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
        self.BBOX_COLOR, self.LINE_WIDTH = (0, 255, 0), 2
        self.KEY_POINT_COLOR, self.KEY_POINT_WIDTH, self.KEY_POINT_RADIUS = (20, 20, 20), 3, 3
