import cv2
import numpy as np

KEY_POINT_COLOR, KEY_POINT_WIDTH, KEY_POINT_RADIUS = (20, 20, 20), 3, 3
SKELETON_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
SKELETON = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
BOX_SHIFT = 0.2
BOX_ENLARGE = 1.3
TARGET_TRIANGLE = np.float32([[128, 128],
                              [128, 0],
                              [0, 128]])


def get_triangle(kp0, kp2, dist=1):
    """get a triangle used to calculate Affine transformation matrix"""

    dir_v = kp2 - kp0
    dir_v /= np.linalg.norm(dir_v)

    dir_v_r = dir_v @ np.r_[[[0, 1], [-1, 0]]].T
    return np.float32([kp2,
                       kp2 + dir_v * dist,
                       kp2 + dir_v_r * dist])


def triangle_to_bbox(source):
    # plain old vector arithmetics
    bbox = np.c_[
        [source[2] - source[0] + source[1]],
        [source[1] + source[0] - source[2]],
        [3 * source[0] - source[1] - source[2]],
        [source[2] - source[1] + source[0]],
    ].reshape(-1, 2)
    return bbox


def get_sources(bboxes, keypoints_set):
    sources = []
    for i in range(len(bboxes)):
        keypoints = keypoints_set[i]
        x, y, w, h = bboxes[i]
        side = max(w, h) * BOX_ENLARGE
        source = get_triangle(keypoints[0], keypoints[2], side)
        source -= (keypoints[0] - keypoints[2]) * BOX_SHIFT
        sources.append(source)
    return sources


def warp_affine(padding_image, source):
    Mtr = cv2.getAffineTransform(source * IMAGE_WIDTH / INPUT_WIDTH, TARGET_TRIANGLE)
    img_landmark = cv2.warpAffine(padding_image, Mtr, (INPUT_WIDTH, INPUT_HEIGHT))
    return img_landmark


def convert_landmark_back(joints, source, padding):
    Mtr = cv2.getAffineTransform(source * IMAGE_WIDTH / INPUT_WIDTH, TARGET_TRIANGLE)
    Mtr = np.pad(Mtr.T, ((0, 0), (0, 1)), constant_values=1, mode='constant').T
    Mtr[2, :2] = 0
    MtrInv = np.linalg.inv(Mtr)
    # projecting keypoints back into original image coordinate space
    kp_orig = (np.pad(joints, ((0, 0), (0, 1)), constant_values=1, mode='constant') @ MtrInv.T)[:, :2]
    kp_orig -= padding[::-1]
    return kp_orig


def draw_landmraks_set(image, landmarks_set):
    for i in range(len(landmarks_set)):
        keypoints = landmarks_set[i]
        for j in range(5):
            for k in range(4):
                cv2.line(image,
                         tuple(keypoints[SKELETON[j][k]].astype('int')),
                         tuple(keypoints[SKELETON[j][k + 1]].astype('int')),
                         SKELETON_COLORS[j],
                         3)

    for keypoints in landmarks_set:
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), KEY_POINT_RADIUS, KEY_POINT_COLOR, KEY_POINT_WIDTH)

    return image
