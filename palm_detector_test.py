import os

import cv2
import tensorflow as tf
from utils.detection_utils import preprocess, extract_bboxes_and_keypoints, draw_bboxes, draw_keypoints_set
import time
from collections import deque

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
WINDOW_NAME = 'MediaPipe Palm Detection'
MODEL_PATH = os.path.join('models', 'palm_detection_without_custom_op.tflite')

fps = deque([], maxlen=100)
if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    cv2.resizeWindow(WINDOW_NAME, IMAGE_WIDTH, IMAGE_HEIGHT)
    model = tf.lite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()
    _, INPUT_HEIGHT, INPUT_WIDTH, _ = model.get_input_details()[0]['shape']
    print(INPUT_WIDTH, INPUT_HEIGHT)
    while True:
        # read a frame
        ret, original_frame = cap.read()

        start_time = time.time()
        palm_input_image, padding_image, padding = preprocess(original_frame, INPUT_WIDTH, INPUT_HEIGHT)

        # inference
        output_details = model.get_output_details()
        model.set_tensor(model.get_input_details()[0]['index'], palm_input_image)
        model.invoke()
        output_reg = model.get_tensor(output_details[0]['index'])[0]
        output_clf = model.get_tensor(output_details[1]['index'])[0, :, 0]

        # convert prediction back to original image
        bboxes, keypoints_set, ori_bboxes, ori_keypoints_set = extract_bboxes_and_keypoints(output_reg, output_clf,
                                                                                            padding, original_frame)

        fps.appendleft(1 / (time.time() - start_time))
        cv2.putText(original_frame,
                    'FPS: ' + str(round(sum(fps) / len(fps), 2)),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        # visualize
        original_frame = draw_bboxes(original_frame, bboxes)
        original_frame = draw_keypoints_set(original_frame, keypoints_set)
        cv2.imshow(WINDOW_NAME, original_frame)

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
