from hand_tracker_config import Config
import numpy as np
import math
import cv2

BENT_THRESHOLD = 480


class GestureClassifier:
    def __init__(self):
        self.config = Config()

    def _calculate_angle(self, landmarks):
        """
        calculate angles for MCP, PIP and DIP for each finger
        :param landmarks: 21 landmarks of a hand
        :return: 3 angles for each fingers, e.g. [[d1,d2,d3],[...,...,...],...]
        """
        angle_list = []
        for finger in self.config.SKELETON:
            landmarks_finger = 256 - landmarks[finger, :]
            finger_angle_list = []
            for i in range(1, 4):
                ori = landmarks_finger[i]
                p1 = landmarks_finger[i - 1]
                p2 = landmarks_finger[i + 1]
                v1 = -(ori - p1)
                v2 = p2 - ori
                unit_v1 = v1 / np.linalg.norm(v1)
                unit_v2 = v2 / np.linalg.norm(v2)
                dot_product = np.dot(unit_v1, unit_v2)
                angle = math.degrees(np.arccos(dot_product))
                finger_angle_list.append(angle)
            angle_list.append(finger_angle_list)
        return angle_list

    def _is_bent(self, angle_list):
        is_bent_list = []
        for finger_angle_list in angle_list:
            angle_sum = sum(finger_angle_list)
            if angle_sum < BENT_THRESHOLD:
                is_bent_list.append(True)
            else:
                is_bent_list.append(False)
        return is_bent_list

    def classify(self, landmarks):
        angle_list = self._calculate_angle(landmarks)
        is_bent_list = self._is_bent(angle_list)
        if is_bent_list == [False, False, False, False, False]:
            return "Palm"
        elif is_bent_list == [True, True, True, True, True]:
            return "Fist"
        elif is_bent_list == [True, False, True, True, True]:
            return "One"
        elif is_bent_list == [True, False, False, True, True]:
            return "Two"
        elif is_bent_list == [True, False, False, False, True]:
            return "Three"
        elif is_bent_list == [True, False, False, False, False]:
            return "Four"
        elif is_bent_list == [True, True, False, False, False]:
            return "OK"
        elif is_bent_list == [False, False, True, True, True]:
            return "Pointer"
        else:
            return "None"

    def draw_text(self, image, left, top, text):
        alpha = 0.3
        text_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
        right, bottom = (left + text_size[0][0] + 40, top + text_size[0][1] + 20)

        sub_img = image[top:bottom, left:right]
        text_bg_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, alpha, text_bg_rect, 1 - alpha, 0)
        image[top:bottom, left:right] = res

        cv2.putText(image, text, (left + 20, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
