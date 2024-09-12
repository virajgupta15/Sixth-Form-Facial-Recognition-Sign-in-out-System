import cv2
class WebcamCapture:
    def __init__(self, source=0, backend=cv2.CAP_MSMF):
        self.cap = cv2.VideoCapture(source, backend)  # setting up webcam feed

    def read_frame(self):
        ret, frame = self.cap.read()  # reading the frame
        return ret, frame

    def release(self):
        self.cap.release()  # stopping webcam feed and closing window
        cv2.destroyAllWindows()