import dlib
import cv2
import numpy as np
import line_profiler
from imutils.face_utils import FaceAligner
from detector import FaceDetector

class FaceRecognition:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    detector = FaceDetector()
    aligner = FaceAligner(predictor = predictor, desiredFaceWidth=160)
    recogniser = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    encodings_data = {}  # encodings stored in the form identifier:encoding
    def __init__(self):
        pass

    def euclidean_distance(self, known_encoding, unknown_encoding):
        # takes in 2 encodings and calculates Euclidean distance
        return np.linalg.norm(known_encoding - unknown_encoding)

    def db_compare_encodings(self, unknown_face):
        pass

    def compare_encodings(self, known_encoding, unknown_encoding, threshold=0.55):
        distance = self.euclidean_distance(known_encoding, unknown_encoding)
        return distance < threshold

    def store_encodings(self, identifier, encoding):
        if identifier not in self.encodings_data:
            self.encodings_data[identifier] = encoding

        print(f"Facial encoding for {identifier} stored successfully.")  # confirmation of data storage

    def get_encoding(self, aligned_face, rect):
        # aligned_face is an image represented as a numpy array
        landmarks = self.predictor(aligned_face, rect)  # use dlib predictor to identify landmarks of face
        # Get the facial encoding for the aligned face
        encoding = self.recogniser.compute_face_descriptor(aligned_face, landmarks,100)
        return encoding

    def align_face(self, image, grey, face_rect):   # aligning the face (rotating/transforming)
        #grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        aligned_face = self.aligner.align(image, grey, face_rect)
        return aligned_face

    def train_new_face(self, image_path):
        # get image, lighting correction, detection, alignment, find landmarks, get encoding.
        face_detector = self.detector
        frame = cv2.imread(image_path)  # RBG FORMAT of the image

        # APPLY GAMMA CORRECTION
        grey_frame = face_detector.auto_correct_brightness(frame)  # now a greyscale frame with brightness corrected
        #print("brightness corrected")

        # DETECT FACES
        faces = face_detector.detect_faces(frame)  # greyscale FORMAT - these are the rectangle objs for the faces locations
        largest_face_rect = faces[face_detector.find_largest_face(faces)]  # face rectangle of largest face
        #print("largest face detected")

        # ALIGN FACE
        frame = self.align_face(frame, grey_frame, largest_face_rect)
        #print("face aligned")

        # DRAW BOUNDING BOX
        grey_face_detection = face_detector.detect_faces(frame)  # detect face again in the aligned face image.
        face_detector.draw_bounding_box(frame, grey_face_detection[0])  # get face rectangle and draw on to RGB frame

        # OUTPUT ENCODING
        encoding = self.get_encoding(frame, grey_face_detection[0])
        #print(encoding)
        #print("length:", len(encoding)) # ensuring that its 128
        """cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        return np.array(encoding)
