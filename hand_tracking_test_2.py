import tensorflow as tf
from hand_tracking_utils import *

WINDOW_NAME = 'MediaPipe Hand Tracking'
PALM_MODEL_PATH = os.path.join('models', 'palm_detection_without_custom_op.tflite')
LANDMARK_MODEL_PATH = os.path.join('models', 'hand_landmark.tflite')
NUM_TRACK_HAND = 2

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    cv2.resizeWindow(WINDOW_NAME, IMAGE_WIDTH, IMAGE_HEIGHT)
    # load palm model
    palm_model = tf.lite.Interpreter(model_path=PALM_MODEL_PATH)
    palm_model.allocate_tensors()
    palm_input_details = palm_model.get_input_details()
    palm_output_details = palm_model.get_output_details()
    # load landmark model
    landmark_model = tf.lite.Interpreter(model_path=LANDMARK_MODEL_PATH)
    landmark_model.allocate_tensors()
    landmark_input_details = landmark_model.get_input_details()
    landmark_output_details = landmark_model.get_output_details()
    # out = cv2.VideoWriter('output.mp4', -1, 5., (640, 480))
    num_valid_hand = 0
    is_first_frame = True
    prev_res = None
    while True:
        # read and preprocess a frame
        _, frame = cap.read()
        input_image, padding_image, padding = preprocess(frame, INPUT_WIDTH, INPUT_HEIGHT)
        if is_first_frame or num_valid_hand < NUM_TRACK_HAND:
            print("Palm Detector Activated!")
            is_first_frame = False
            output_reg, output_clf = detect_palm(input_image, palm_model, palm_input_details, palm_output_details)
            key_point_list, side_list = get_res_from_palm_detector(output_reg, output_clf)
        else:
            print("Palm Detector Not Activated!")
            key_point_list, side_list = get_res_from_prev_res(prev_res)

        prev_res = []
        for i in range(len(key_point_list)):
            hand, source = get_hand(padding_image, key_point_list[i], side_list[i])
            landmark, is_hand = detect_landmark(hand, landmark_model, landmark_input_details, landmark_output_details)
            if is_hand:
                landmark, landmark_input = convert_landmark_back(landmark, source, padding, frame)
                frame = draw_landmark(frame, landmark)
                prev_res.append(landmark_input)

        num_valid_hand = len(prev_res)
        cv2.imshow(WINDOW_NAME, frame)
        # out.write(original_frame)

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
