import dlib
import cv2

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def find_largest_face(self, facesArray):
        largest_face_indx = max(range(len(facesArray)), key=lambda i:facesArray[i].area())
        return largest_face_indx

    def draw_bounding_box(self, frame, face):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces

class WebcamCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    face_detector = FaceDetector()
    webcam = WebcamCapture()

    while True:
        ret, frame = webcam.read_frame()

        if ret:
            frame = cv2.flip(frame, 1)
            faces = face_detector.detect_faces(frame)

            if len(faces) > 0:
                largest_face_indx = face_detector.find_largest_face(faces)
                face_detector.draw_bounding_box(frame, faces[largest_face_indx])

            cv2.imshow('Facial Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()

if __name__ == "__main__":
    main()
