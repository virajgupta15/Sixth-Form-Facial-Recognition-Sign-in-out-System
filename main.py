import dlib
import cv2

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def find_largest_face(self, facesArray):  # finding the largest face in image based on area
        largest_face_indx = max(range(len(facesArray)), key=lambda i:facesArray[i].area())
        return largest_face_indx  # returning index of the largest face in the array.

    def draw_bounding_box(self, frame, face):  # drawing the bounding box around face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()  # coordinates of face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces

class WebcamCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # setting up webcam feed

    def read_frame(self):
        ret, frame = self.cap.read()  # reading the frame
        return ret, frame

    def release(self):
        self.cap.release()  # stopping webcam feed and closing window
        cv2.destroyAllWindows()

def main():
    face_detector = FaceDetector()  # instantiating detector object from class
    webcam = WebcamCapture()  # starting the webcam capture

    while True:
        ret, frame = webcam.read_frame()  # reading a frame, ret is a bool value indicating success status

        if ret:  # if frame successfully read
            frame = cv2.flip(frame, 1)  # mirror the image
            faces = face_detector.detect_faces(frame)  # call the detector

            if len(faces) > 0:
                largest_face_indx = face_detector.find_largest_face(faces)  # finding the largest face in frame
                face_detector.draw_bounding_box(frame, faces[largest_face_indx])  # draw bounding box

            cv2.imshow('Facial Detection', frame)  # displaying window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # break out of webcam feed - for testing
            break

    webcam.release()  # stopping webcam feed

if __name__ == "__main__":
    main()
