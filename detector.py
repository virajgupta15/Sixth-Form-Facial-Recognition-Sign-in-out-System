import dlib
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def find_largest_face(self, facesArray):  # finding the largest face in image based on area
        # facesArray is an array of the dlib face rectangle objects.
        largest_face_indx = max(range(len(facesArray)), key=lambda i:facesArray[i].area())
        return largest_face_indx  # returning index of the largest face RECTANGLE in the array.

    def rect_to_tuple(rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()

    def draw_bounding_box(self, frame, face):  # drawing the bounding box around face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()  # coordinates of face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def detect_faces(self, frame):
        try:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # attempt to convert the frame to greyscale unless already in that format.
        except:
            grey = frame
        faces = self.detector(grey)
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
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(frame)
        # print(f"original brightness {avg_brightness}")  # for testing
        # Adjust gamma based on average brightness
        if avg_brightness > 160:
            gamma = 0.8  # Increase gamma for bright images to make them darker
            #print("Too bright - Making image darker")

        elif 90 > avg_brightness > 30:
            gamma = 1.43 # Decrease gamma for dark images to make them brighter
            #print("Too dark - making image brighter")

        elif avg_brightness < 30:
            gamma = 1.7  # Decrease gamma for dark images to make them brighter
            #print("Extremely dark - Making image brighter")

        else:
            #print("no adjustment")  # for testing
            return grey_frame  # greyscale frame

        return self.adjust_gamma(grey_frame, gamma)  # returns a GREY image