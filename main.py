import dlib
import cv2
import numpy as np

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
    def adjust_gamma(self, frame, gamma=1.0):  # gamma is an optional parameter - if not provided, default value = 1.0
        # build a lookup table mapping each pixel to its adjusted gamma values
        # lookup table speeds up the process - calculates the adjusted gamma for each pixel and stores it in memory.
        # gamma calculation : Output pixel value = inputValue ** gamma
        # calculating exponentials is time-consuming hence lookup table is better.
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # print(np.mean(frame))
        return cv2.LUT(frame, table) # applying the gamma correction on the frame using the Lookup table

    def auto_correct_brightness(self, frame):
        # first calculate the average brightness of the image
        # then adjust the gamma based on the thresholds (needs to be brighter or darker)
        # call the adjust_gamma function to do this based on the requirements
        avg_brightness = np.mean(frame)
        print(f"original brightness {avg_brightness}")  # for testing
        # Adjust gamma based on average brightness
        if avg_brightness > 160:
            gamma = 0.8  # Increase gamma for bright images to make them darker
            print("Too bright - Making image darker")

        elif 90 > avg_brightness > 30:
            gamma = 1.43 # Decrease gamma for dark images to make them brighter
            print("Too dark - making image brighter")

        elif avg_brightness < 30:
            gamma = 1.7  # Decrease gamma for dark images to make them brighter
            print("Extremely dark - Making image brighter")

        else:
            print("no adjustment")  # for testing
            return frame

        return self.adjust_gamma(frame, gamma)


class FaceRecognition():
    def __init__(self):
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def facial_landmarks(self, face):
        pass
    def train_new_face(self, image_path, identifier):
        # get image, lighting correction, detection, crop to face, find landmarks, get encoding.
        face_detector = FaceDetector()  # Create an instance of the FaceDetector class

        image = cv2.imread(image_path)
        new_frame = face_detector.auto_correct_brightness(image)  # Apply gamma correction
        print("brightness corrected")   
        faces = face_detector.detect_faces(new_frame)  # Detect faces
        print("face detected")


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
            new_frame = face_detector.auto_correct_brightness(frame)
            # print(np.mean(new_frame))
            faces = face_detector.detect_faces(new_frame)  # call the detector

            if len(faces) > 0:
                largest_face_index = face_detector.find_largest_face(faces)  # finding the largest face in frame
                face_detector.draw_bounding_box(new_frame, faces[largest_face_index])  # draw bounding box

            cv2.imshow('Facial Detection', new_frame)  # displaying window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # break out of webcam feed - for testing
            break

    webcam.release()  # stopping webcam feed

if __name__ == "__main__":
    main()

face_recogniser = FaceRecognition()
FaceRecognition.train_new_face(image_path="barack obama.jpg",identifier="obama")
