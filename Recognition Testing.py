from recogniser import FaceRecognition
from detector import FaceDetector
from database import DatabaseHandler
import cv2
import dlib

# testing recognition ability - 96%
# testing detection ability - 98%
db = DatabaseHandler("localhost", "facial_recognition_manager",
                     "Faces123!", "facial_recognition_db")
import os

known_faces = db.fetch_encoding_data()
print(known_faces)


def test_face_recognition():
    num_tests = 0
    num_passes = 0
    face_detector = FaceDetector()  # my custom facial recognition class
    detector = dlib.get_frontal_face_detector()  # dlib face detector
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Detects landmarks
    face_recognition = FaceRecognition()
    with open(r"C:\Users\viraj\PycharmProjects\Facial recognition coursework\Data\fetched_encoding_data.txt"
            , "r") as file:
        encodings_data = dict(eval(file.read()))
        # print(encodings_data)  # the facial encodings stored in dictionary personID:encoding
    folder_num = 0
    names = open(r"C:\Users\viraj\PycharmProjects\Facial recognition coursework\Test images\list_name.txt",
                 "r").readlines()
    trained_face_paths = [i[:-1] for i in open(
        r"C:\Users\viraj\PycharmProjects\Facial recognition coursework\Test images\bestFaces.txt", "r").readlines()]
    print(trained_face_paths)
    root_dir = r'C:\Users\viraj\Downloads\cfp-dataset\cfp-dataset\Data\Images'

    for person_folder in os.listdir(root_dir):
        expected_person = names[folder_num]
        print("expected person:", expected_person)
        person_path = os.path.join(root_dir, person_folder)
        folder_num += 1
        if os.path.isdir(person_path):
            # Access the frontal images folder for the person
            frontal_folder = os.path.join(person_path, 'frontal')
            if os.path.exists(frontal_folder) and os.path.isdir(frontal_folder):
                # Iterate over images within the frontal folder
                for image_file in os.listdir(frontal_folder):
                    best_match_distance = float(
                        'inf')  # Initialise with infinity to ensure any valid match will have a smaller distance
                    best_match_identifier = None  # Initialize with None to track the identifier of the best match
                    image_path = os.path.join(frontal_folder, image_file)
                    try:
                        img = dlib.load_rgb_image(image_path)
                    except:
                        continue
                    dets = detector(img, 1)  # Detect faces in the image
                    if len(dets) > 0:
                        shape = predictor(img, dets[0])  # Get facial landmarks for the first detected face
                        if all([shape.part(i) for i in range(68)]):  # All facial landmarks are detected
                            try:
                                if image_path not in trained_face_paths:
                                    # ensuring that the current image was not used for training
                                    unknown_encoding = face_recognition.train_new_face(image_path)
                                    # Compare the face encoding with known faces in the database
                                    for identifier, known_encoding in known_faces.items():
                                        # identifier is the index = studentID
                                        is_match, distance = face_recognition.compare_encodings(
                                            known_encoding=known_encoding,
                                            unknown_encoding=unknown_encoding)
                                        if is_match and distance < best_match_distance:
                                            # Update the best match if the current match is better (lower distance)
                                            best_match_distance = distance
                                            best_match_identifier = identifier

                                    # Check if any match was found
                                    if best_match_identifier is not None:
                                        # Match found, update face ID label with the student's name
                                        match = db.get_student_name(best_match_identifier)
                                        print(f"Best match is: {match} with distance: {best_match_distance}")
                                        print(expected_person[:-1])
                                        if match == expected_person[:-1]:
                                            num_passes += 1

                                        print(image_path)
                                    else:
                                        print("No match found.")
                                    num_tests += 1
                                    print(num_tests, num_passes)

                            except:
                                continue

    # RESULTS
    print(f"No. tests: {num_tests}")
    print(f"No. passed test cases: {num_passes}")
    print(f"Percentage accuracy: {(num_passes / num_tests) * 100}%")


def test_face_detector():
    num_tests = 0
    num_passes = 0
    face_detector = FaceDetector()  # my custom facial recognition class
    detector = dlib.get_frontal_face_detector()  # dlib face detector
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Detects landmarks
    root_dir = r'C:\Users\viraj\Downloads\cfp-dataset\cfp-dataset\Data\Images'
    # Iterate over each individual's folder
    for person_folder in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person_folder)
        if os.path.isdir(person_path):
            # Access the frontal images folder for the person
            frontal_folder = os.path.join(person_path, 'frontal')
            if os.path.exists(frontal_folder) and os.path.isdir(frontal_folder):
                # Iterate over images within the frontal folder
                for image_file in os.listdir(frontal_folder):
                    image_path = os.path.join(frontal_folder, image_file)

                    # First check if valid image - landmarks are all visible
                    try:
                        img = dlib.load_rgb_image(image_path)
                    except:
                        continue
                    # Detect faces in the image
                    dets = detector(img, 1)
                    if len(dets) > 0:
                        # Get facial landmarks for the first detected face
                        shape = predictor(img, dets[0])
                        if all([shape.part(i) for i in range(68)]):
                            # All facial landmarks are detected
                            num_tests += 1

                            try:
                                frame = cv2.imread(image_path)

                                # Try my custom facial detection pipeline
                                corrected_frame = face_detector.auto_correct_brightness(frame)
                                face = face_detector.detect_faces(corrected_frame)
                                # Draw bounding boxes around detected faces
                                face_detector.draw_bounding_box(corrected_frame, face[0])
                                num_passes += 1
                            except:
                                print("Facial detection failed for this image", image_path)
                        else:
                            print("Not a valid face")

    # RESULTS
    print(f"No. tests: {num_tests}")
    print(f"No. passed test cases: {num_passes}")
    print(f"Percentage accuracy: {(num_passes / num_tests) * 100}%")


test_face_recognition()
