import shutil
import cv2
import dlib
import numpy as np
import os

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def assess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        # print(image_path)
        return 0
    landmarks = shape_predictor(gray, faces[0])
    x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
    roi = img[y:y + h, x:x + w]

    # Check if the ROI is not empty before calculating the mean
    if roi.size == 0:
        return 0
    # Calculate the average pixel intensity in the ROI
    mean_brightness = np.mean(roi)
    # true=1 if all facial features are visible
    all_landmarks_visible = all(landmarks.part(i).x > 0 and landmarks.part(i).y > 0 for i in range(68))
    # Combine metrics
    if mean_brightness > 0:
        quality_score = mean_brightness * all_landmarks_visible
        return quality_score
    else:
        return 0

# Function to find the best image for each person
def find_best_images(folder_path):
    best_images = {}
    best_score = 0
    for celebrity_folder in os.listdir(folder_path):
        celebrity_path = os.path.join(folder_path, celebrity_folder)

        if os.path.isdir(celebrity_path):
            best_image = None
            best_score = 0

            for image_file in os.listdir(celebrity_path):
                image_path = os.path.join(celebrity_path, image_file)

                # Assess each image and get quality score
                quality_score = assess_image(image_path)

                if quality_score > best_score:
                    best_score = quality_score
                    best_image = image_path

            if best_image is not None:
                best_images[celebrity_folder] = best_image

    if best_score == 0:
        print(folder_path, "no good images")

    return best_images

def delete_folder(folder_path, folder_to_delete):
    for celebrity_folder in os.listdir(folder_path):
        celebrity_path = os.path.join(folder_path, celebrity_folder)

        if os.path.isdir(celebrity_path):
            folder_to_delete_path = os.path.join(celebrity_path, folder_to_delete)

            # Check if the folder to delete exists
            if os.path.exists(folder_to_delete_path):
                try:
                    # Delete the folder and its contents
                    shutil.rmtree(folder_to_delete_path)
                    print(f"Deleted {folder_to_delete} for {celebrity_folder}")
                except Exception as e:
                    print(f"Error deleting {folder_to_delete} for {celebrity_folder}: {e}")

"""def x():
    for i in range(1, 501):
        num = str(i).zfill(3)

        folder_path = rf"C://Users//viraj//Downloads//cfp-dataset//cfp-dataset//Data//Images//{num}"
        folder_to_delete = fr"C://Users//viraj//Downloads//cfp-dataset//cfp-dataset//Data//Images//{num}//profile"
        delete_folder(folder_path, folder_to_delete)"""

with open("../Test images/bestFaces.txt", "w") as f:
    for i in range(1,501):
        num = str(i).zfill(3)  # format 000
        path = find_best_images(fr"C:\Users\viraj\Downloads\cfp-dataset\cfp-dataset\Data\Images\{num}")
        f.write(str(path.get("frontal"))+"\n")

