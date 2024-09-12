"""import dlib
import cv2
import numpy as np
import json
from detector import FaceDetector
from recogniser import FaceRecognition
from webcam import WebcamCapture
from database import DatabaseHandler

face_recogniser = FaceRecognition()
face_detector = FaceDetector()  # instantiating detector object from class
webcam = WebcamCapture()  # starting the webcam capture

"""old_known_faces = {
"Donald Trump": face_recogniser.train_new_face("Testing known faces/Donald Trump.jpg"),
"Jackie Chan": face_recogniser.train_new_face("Testing known faces/Jackie Chan.jpg"),
"Joe Biden": face_recogniser.train_new_face("Testing known faces/Joe Biden.jpg"),
"Narendra Modi": face_recogniser.train_new_face("Testing known faces/Narendra Modi.jpg"),
"Will Smith": face_recogniser.train_new_face("Testing known faces/Will Smith.jpg")}"""

def main():
    face_roi = None
    count = 1
    while True:
        count += 1
        ret, frame = webcam.read_frame()  # reading a frame (BGR), ret is a bool value indicating success status
        if ret:  # if frame successfully read
            frame = cv2.flip(frame, 1)  # mirror the image
            frame = cv2.resize(frame, (0,0), 5, fx=0.5, fy=0.5) # resizing the image
            grey_frame = face_detector.auto_correct_brightness(frame=frame, face_roi=face_roi)
            faces = face_detector.detect_faces(grey_frame)

            if len(faces) > 0:  # if a face is detected
                largest_face = faces[face_detector.find_largest_face(faces)]  # finding the location of largest face
                face_roi = face_detector.rect_to_tuple(largest_face)

                 # this is NOT greyscale after update
                #print("roi lighting checked")

                if count % 25 == 0:
                    aligned_face = face_recogniser.align_face(frame, grey_frame, face_rect=largest_face)
                    cv2.imshow("aligned face", aligned_face)
                    if cv2.waitKey(1) & 0xFF == ord('w'):  # break out of webcam feed - for testing
                        break

                    try:
                        # getting encoding of aligned face from webcam
                        encoding = face_recogniser.get_encoding(aligned_face, face_detector.detect_faces(aligned_face)[0])
                        json_encoding = json.dumps(list(encoding))  # CONVERTING THE ENCODING INTO JSON
                        for identifier, known_encoding in known_faces.items():
                            if face_recogniser.compare_encodings(known_encoding,encoding):
                                print(f"match is: {db.get_student_name(identifier)}")
                                break
                    except IndexError:
                        pass
                face_detector.draw_bounding_box(frame, largest_face)  # draw bounding box
            else:
                face_roi = None
            cv2.imshow('Webcam Feed', frame)  # displaying window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # break out of webcam feed - for testing
            break

    webcam.release()  # stopping webcam feed

if __name__ == "__main__":
    db = DatabaseHandler("localhost", "facial_recognition_manager", "Faces123!", "facial_recognition_db")
    known_faces = db.fetch_encoding_data()  # a dictionary of StudentID:FacialEncoding fetched from SQL server
    
    print("main2")

    #db.sign_in_out_student(student_id=1, is_sign_in=False)  #SIGN OUT

    main()
"""    with open("../Data/fetched_encoding_data.txt", "r") as f:
        for line in f:
            dictionary = json.loads(line)
            # Convert the 'facial_encoding' key from a list back to a NumPy array
            if 'facial_encoding' in dictionary:
                dictionary['facial_encoding'] = np.array(dictionary['facial_encoding'])
            known_faces[dictionary['name']] = dictionary['facial_encoding']""""""