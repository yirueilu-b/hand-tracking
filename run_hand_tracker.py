import cv2
from utils.hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
WINDOW_NAME = "Hand Tracking Demo"

if __name__ == '__main__':
    hand_tracker = HandTracker()
    gesture_classifier = GestureClassifier()
    cap = cv2.VideoCapture(1)
    cv2.resizeWindow(WINDOW_NAME, hand_tracker.config.IMAGE_WIDTH, hand_tracker.config.IMAGE_HEIGHT)
    out = cv2.VideoWriter('output.mp4', -1, 5., (640, 480))
    while True:
        _, frame = cap.read()
        landmarks_list = hand_tracker.inference(frame)
        for landmarks in landmarks_list: hand_tracker.draw_landmark(frame, landmarks)

        gesture = "None"
        if landmarks_list:
            landmarks = landmarks_list[0]
            gesture = gesture_classifier.classify(landmarks)
        gesture_classifier.draw_text(frame, 0, 0, gesture)

        cv2.imshow(WINDOW_NAME, frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
