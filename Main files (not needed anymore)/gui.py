from tkinter import *
from ttkbootstrap import *
from PIL import Image, ImageTk
import cv2
import json
from detector import FaceDetector
from recogniser import FaceRecognition
from webcam import WebcamCapture
from database import DatabaseHandler

class FacialRecognitionPage:
    def __init__(self, root):
        # Initialize the Tkinter root window and set its size
        self.root = root
        self.root.geometry("1280x1240")
        self.root.title("Facial Recognition Page")

        # Create a label to display the video feed
        self.video_label = ttk.Label(self.root)
        self.video_label.place(x=10, y=10)

        # Label to display face identity
        self.face_id_label = ttk.Label(self.root, text="Identity:", foreground="red")
        self.face_id_label.place(x=10, y=310)

        # Label for alternative identification
        self.alt_id_label = ttk.Label(self.root, text="Alternative Identification:")
        self.alt_id_label.place(x=450, y=10)

        # Label and entry for username
        self.user_name_label = ttk.Label(self.root, text="Username:")
        self.user_name_label.place(x=450, y=50, width=70, height=25)
        self.user_name_entry = ttk.Entry(self.root, validate="key",
                                         validatecommand=(self.root.register(self.input_length_restrictor), "%P", 10))
        self.user_name_entry.place(x=530, y=50, width=160, height=30)

        # Label and entry for password
        self.password_label = ttk.Label(self.root, text="Password:")
        self.password_label.place(x=450, y=110, width=70, height=25)
        self.password_entry = ttk.Entry(self.root, show="*",
                                        validate="key",
                                        validatecommand=(self.root.register(self.input_length_restrictor), "%P", 30))

        self.password_entry.place(x=530, y=110, width=160, height=30)

        self.error_label = ttk.Label(self.root, text="", foreground="red")
        self.error_label.place(x=530, y=190, width=200, height=30)

        # Button to submit credentials
        self.enter_button = ttk.Button(self.root, text="Enter", command=self.validate_username_password)
        self.enter_button.place(x=530, y=160, width=70, height=25)

        # Initialize variables for face recognition
        self.face_roi = None
        self.count = 1

        # Initialize face recognition components
        self.face_recogniser = FaceRecognition()
        self.face_detector = FaceDetector()
        self.webcam = WebcamCapture()

        # Connect to the database and fetch known faces
        self.db = DatabaseHandler("localhost", "facial_recognition_manager", "Faces123!", "facial_recognition_db")
        self.known_faces = self.db.fetch_encoding_data()

        # Start the update loop
        self.update()
    def update(self):
        # Update function to capture frames and perform face recognition
        self.count += 1
        ret, frame = self.webcam.read_frame()

        if ret:
            # Preprocess the frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (0, 0), 5, fx=0.5, fy=0.5)
            grey_frame = self.face_detector.auto_correct_brightness(frame=frame, face_roi=self.face_roi)
            faces = self.face_detector.detect_faces(grey_frame)

            if len(faces) == 0:
                self.face_id_label.config(text="Identity: Unknown", foreground="red")
            elif len(faces) > 0:
                # If a face is detected, process it
                largest_face = faces[self.face_detector.find_largest_face(faces)]
                self.face_roi = self.face_detector.rect_to_tuple(largest_face)

                if self.count % 25 == 0:
                    # Align the face for recognition
                    aligned_face = self.face_recogniser.align_face(frame, grey_frame, face_rect=largest_face)
                    cv2.imshow("aligned face", aligned_face)

                    try:
                        # Perform face recognition
                        encoding = self.face_recogniser.get_encoding(aligned_face,
                                                                     self.face_detector.detect_faces(aligned_face)[0])
                        json_encoding = json.dumps(list(encoding))

                        # Loop through known faces to find a match
                        for identifier, known_encoding in self.known_faces.items():
                            if self.face_recogniser.compare_encodings(known_encoding, encoding):
                                # Match found, get the student's name from the database
                                match = self.db.get_student_name(identifier)
                                print(f"Match is: {match}")

                                # Update GUI label with the identified name and set color to green
                                self.face_id_label.config(text=f"Identity: {match}", foreground="green")
                                break  # Exit the loop when a match is found
                            else:
                                # If no match is found, update GUI label as "Unknown" and set color to red
                                self.face_id_label.config(text="Identity: Unknown", foreground="red")

                    except IndexError:
                        # If no match is found, update GUI label as "Unknown" and set color to red
                        self.face_id_label.config(text="Identity: Unknown", foreground="red")
                        pass  # Handle index error, if any

                # Draw bounding box around the face
                self.face_detector.draw_bounding_box(frame, largest_face)
            else:
                self.face_roi = None

            # Convert frame to RGB format for display in Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Call the update function after a delay (10 milliseconds)
            self.video_label.after(10, self.update)

    def validate_username_password(self):
        username = self.user_name_entry.get()
        password = self.password_entry.get()
        #print(self.db.fetch_password(username)[0])
        db_password = self.db.fetch_password(username)
        if db_password is None or password != db_password[0]:
            self.error_label.config(text="Incorrect username or password")
            return False
        else:  # valid credentials
            self.error_label.config(text="") # reset error label text
            return True

    def input_length_restrictor(self, new_value, max_length):
        # This function is called for every keypress to validate the input size if it affects the desired object
        return len(new_value) <= int(max_length)



if __name__ == "__main__":
    # Create the Tkinter root window and start the application
    root = Tk()
    app = FacialRecognitionPage(root)
    root.mainloop()