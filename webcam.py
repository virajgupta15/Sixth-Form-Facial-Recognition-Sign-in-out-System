import cv2
class WebcamCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # setting up webcam feed

    def read_frame(self):
        ret, frame = self.cap.read()  # reading the frame
        return ret, frame

    def release(self):
        self.cap.release()  # stopping webcam feed and closing window
        cv2.destroyAllWindows()