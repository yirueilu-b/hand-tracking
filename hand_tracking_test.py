import os

import cv2
import tensorflow as tf
from utils.detection_utils import preprocess, extract_bboxes_and_keypoints, draw_bboxes, draw_keypoints_set, normalize_image
from utils.landmark_utils import get_sources, warp_affine, convert_landmark_back, draw_landmraks_set

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
WINDOW_NAME = 'MediaPipe Hand Tracking'
PALM_MODEL_PATH = os.path.join('models', 'palm_detection_without_custom_op.tflite')
LANDMARK_MODEL_PATH = os.path.join('models', 'hand_landmark.tflite')

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    cv2.resizeWindow(WINDOW_NAME, IMAGE_WIDTH, IMAGE_HEIGHT)
    # load models
    palm_model = tf.lite.Interpreter(model_path=PALM_MODEL_PATH)
    palm_model.allocate_tensors()
    palm_input_details = palm_model.get_input_details()
    palm_output_details = palm_model.get_output_details()
    landmark_model = tf.lite.Interpreter(model_path=LANDMARK_MODEL_PATH)
    landmark_model.allocate_tensors()
    landmark_input_details = landmark_model.get_input_details()
    landmark_output_details = landmark_model.get_output_details()
    # out = cv2.VideoWriter('output.mp4', -1, 5., (640, 480))

    print(INPUT_WIDTH, INPUT_HEIGHT)
    while True:
        # read and preprocess a frame
        ret, original_frame = cap.read()
        palm_input_image, padding_image, padding = preprocess(original_frame, INPUT_WIDTH, INPUT_HEIGHT)
        # inference palm model
        palm_model.set_tensor(palm_input_details[0]['index'], palm_input_image)
        palm_model.invoke()
        output_reg = palm_model.get_tensor(palm_output_details[0]['index'])[0]
        output_clf = palm_model.get_tensor(palm_output_details[1]['index'])[0, :, 0]
        # convert prediction back to original image
        bboxes, keypoints_set, ori_bboxes, ori_keypoints_set = extract_bboxes_and_keypoints(output_reg, output_clf,
                                                                                            padding)
        # visualize palm
        original_frame = draw_bboxes(original_frame, bboxes)
        original_frame = draw_keypoints_set(original_frame, keypoints_set)
        # inference landmark model
        sources = get_sources(ori_bboxes, ori_keypoints_set)
        landmarks_set = []
        for i in range(len(ori_bboxes)):
            source = sources[i]
            landmark_input_image = normalize_image(warp_affine(padding_image, source))
            landmark_model.set_tensor(landmark_input_details[0]['index'], landmark_input_image.reshape(1, 256, 256, 3))
            landmark_model.invoke()
            joints = landmark_model.get_tensor(landmark_output_details[0]['index'])
            landmarks_set.append(convert_landmark_back(joints.reshape(-1, 2), source, padding))
        # visualize landmarks
        original_frame = draw_landmraks_set(original_frame, landmarks_set)
        cv2.imshow(WINDOW_NAME, original_frame)
        # out.write(original_frame)

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
