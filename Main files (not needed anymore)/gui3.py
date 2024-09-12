from tkinter import *
from ttkbootstrap import *
from PIL import Image, ImageTk
import cv2
import json
from detector import FaceDetector
from recogniser import FaceRecognition
from webcam import WebcamCapture
from database import DatabaseHandler

class BasePage:
    def __init__(self, root, switch_page_callback):
        self.root = root
        self.switch_page_callback = switch_page_callback

    def switch_to_page(self, page_class):
        if hasattr(self, 'frame'):
            self.frame.destroy()
        self.frame = page_class(self.root, self.switch_to_page)


class FacialRecognitionPage(BasePage):
    def __init__(self, root, switch_page_callback):
        super().__init__(root, switch_page_callback)
        self.root.geometry("1280x1240")
        self.root.title("Facial Recognition Page")
        self.identified = False

        self.video_label = ttk.Label(self.root)
        self.video_label.place(x=10, y=10)

        self.face_id_label = ttk.Label(self.root, text="Identity:", foreground="red")
        self.face_id_label.place(x=10, y=310)

        self.alt_id_label = ttk.Label(self.root, text="Alternative Identification:")
        self.alt_id_label.place(x=450, y=10)

        self.user_name_label = ttk.Label(self.root, text="Username:")
        self.user_name_label.place(x=450, y=50, width=70, height=25)
        self.user_name_entry = ttk.Entry(self.root, validate="key",
                                         validatecommand=(self.root.register(self.input_length_restrictor), "%P", 10))
        self.user_name_entry.place(x=530, y=50, width=156, height=30)

        self.password_label = ttk.Label(self.root, text="Password:")
        self.password_label.place(x=450, y=110, width=70, height=25)
        self.password_entry = ttk.Entry(self.root, show="*",
                                        validate="key",
                                        validatecommand=(self.root.register(self.input_length_restrictor), "%P", 30))
        self.password_entry.place(x=530, y=110, width=156, height=30)

        self.error_label = ttk.Label(self.root, text="", foreground="red")
        self.error_label.place(x=530, y=190, width=200, height=30)

        self.enter_button = ttk.Button(self.root, text="Enter", command=self.validate_username_password)
        self.enter_button.place(x=530, y=160, width=70, height=25)

        # attributes for recognition algorithm
        self.face_roi = None
        self.count = 1

        self.face_recogniser = FaceRecognition()
        self.face_detector = FaceDetector()
        self.webcam = WebcamCapture()

        self.db = DatabaseHandler("localhost", "facial_recognition_manager", "Faces123!", "facial_recognition_db")
        self.known_faces = self.db.fetch_encoding_data()

        self.update()  # refreshing the page

    def update(self):
        self.count += 1
        ret, frame = self.webcam.read_frame()

        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (0, 0), 5, fx=0.5, fy=0.5)
            grey_frame = self.face_detector.auto_correct_brightness(frame=frame, face_roi=self.face_roi)
            faces = self.face_detector.detect_faces(grey_frame)

            if len(faces) == 0:
                self.identified = False
                self.face_id_label.config(text="Identity: Unknown", foreground="red")
            elif len(faces) > 0:
                largest_face = faces[self.face_detector.find_largest_face(faces)]
                self.face_roi = self.face_detector.rect_to_tuple(largest_face)

                if self.count % 25 == 0:
                    aligned_face = self.face_recogniser.align_face(frame, grey_frame, face_rect=largest_face)
                    cv2.imshow("aligned face", aligned_face)

                    try:
                        encoding = self.face_recogniser.get_encoding(aligned_face,
                                                                     self.face_detector.detect_faces(aligned_face)[0])
                        json_encoding = json.dumps(list(encoding))

                        for identifier, known_encoding in self.known_faces.items():
                            if self.face_recogniser.compare_encodings(known_encoding, encoding):
                                match = self.db.get_student_name(identifier)
                                print(f"Match is: {match}")
                                self.face_id_label.config(text=f"Identity: {match}", foreground="green")
                                self.identified = True
                                break
                            else:
                                self.identified = False
                                self.face_id_label.config(text="Identity: Unknown", foreground="red")


                    except IndexError:
                        self.face_id_label.config(text="Identity: Unknown", foreground="red")
                        pass

                self.face_detector.draw_bounding_box(frame, largest_face)
            else:
                self.face_roi = None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.update)

    def validate_username_password(self):
        username = self.user_name_entry.get()
        password = self.password_entry.get()
        db_password = self.db.fetch_password(username)
        if db_password is None or password != db_password[0]:
            self.error_label.config(text="Incorrect username or password")
            return False
        else:
            self.error_label.config(text="")
            self.switch_page_callback()  # Switch to the next page
            return True

    def input_length_restrictor(self, new_value, max_length):
        return len(new_value) <= int(max_length)


class SignInOutPage(BasePage):
    def __init__(self, root, switch_page_callback):
        super().__init__(root, switch_page_callback)
        self.root.geometry("800x600")
        self.root.title("Sign In/Out Page")

        # Button to switch to Facial Recognition page
        self.switch_button = ttk.Button(self.root, text="Back to Identification Page", command=lambda: self.switch_to_page(FacialRecognitionPage))



class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Main Application")


if __name__ == "__main__":
    root = Tk()
    app = MainApplication(root)
    root.mainloop()



class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Main Application")
        self.current_page = None
        self.switch_to_facial_recognition()

    def switch_page(self, PageClass):
        # Clear the current page's content
        if self.current_page:
            self.current_page.destroy()

        # Create an instance of the new page
        self.current_page = PageClass(self.root, self.switch_page)

    def switch_to_facial_recognition(self):
        self.switch_page(FacialRecognitionPage)

    def switch_to_sign_in_out(self):
        self.switch_page(SignInOutPage)

if __name__ == "__main__":
    root = Tk()
    app = MainApplication(root)
    root.mainloop()