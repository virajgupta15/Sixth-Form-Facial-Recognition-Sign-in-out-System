import dlib
import cv2
from detector import FaceDetector
from recogniser import FaceRecognition
from webcam import WebcamCapture
from database import DatabaseHandler

face_recogniser = FaceRecognition()
face_detector = FaceDetector()  # instantiating detector object from class
webcam = WebcamCapture()  # starting the webcam capture

known_faces = {"Donald Trump": face_recogniser.train_new_face("Testing known faces/Donald Trump.jpg"),
               "Jackie Chan": face_recogniser.train_new_face("Testing known faces/Jackie Chan.jpg"),
               "Joe Biden": face_recogniser.train_new_face("Testing known faces/Joe Biden.jpg"),
               "Narendra Modi": face_recogniser.train_new_face("Testing known faces/Narendra Modi.jpg"),
               "Will Smith": face_recogniser.train_new_face("Testing known faces/Will Smith.jpg")}


def main():

    print("hello")
    count = 1
    while True:
        count += 1
        ret, frame = webcam.read_frame()  # reading a frame (BGR), ret is a bool value indicating success status
        if ret:  # if frame successfully read
            frame = cv2.flip(frame, 1)  # mirror the image
            frame = cv2.resize(frame, (0,0), 5, fx=0.5, fy=0.5) # resizing the image
            grey_frame = face_detector.auto_correct_brightness(frame)  # this is greyscale
            faces = face_detector.detect_faces(grey_frame)  # call the detector and detect the faces

            if len(faces) > 0:
                largest_face = faces[face_detector.find_largest_face(faces)] # finding the largest face in frame

                if count % 25 == 0:
                    aligned_face = face_recogniser.align_face(frame, grey_frame, largest_face)
                    cv2.imshow("webcam capture",aligned_face)
                    if cv2.waitKey(1) & 0xFF == ord('w'):  # break out of webcam feed - for testing
                        break

                    try:
                        # getting encoding of aligned face from webcam
                        encoding = face_recogniser.get_encoding(aligned_face, face_detector.detect_faces(aligned_face)[0])
                        print(encoding)

                        for identifier, known_encoding in known_faces.items():
                            if face_recogniser.compare_encodings(known_encoding,encoding):
                                print(f"match is: {identifier}")
                                break
                    except IndexError:
                        pass

                face_detector.draw_bounding_box(frame, largest_face)  # draw bounding box
            cv2.imshow('Facial Detection', frame)  # displaying window


        if cv2.waitKey(1) & 0xFF == ord('q'):  # break out of webcam feed - for testing
            break

    webcam.release()  # stopping webcam feed

if __name__ == "__main__":
    main()




