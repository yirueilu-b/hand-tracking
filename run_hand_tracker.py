import cv2
from utils.hand_tracker import HandTracker

WINDOW_NAME = "Hand Tracking Demo"

if __name__ == '__main__':
    handTracker = HandTracker()
    cap = cv2.VideoCapture(1)
    cv2.resizeWindow(WINDOW_NAME, handTracker.config.IMAGE_WIDTH, handTracker.config.IMAGE_HEIGHT)
    while True:
        _, frame = cap.read()
        landmarks_list = handTracker.inference(frame)
        for landmarks in landmarks_list: handTracker.draw_landmark(frame, landmarks)
        cv2.imshow(WINDOW_NAME, frame)
        # out.write(original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
