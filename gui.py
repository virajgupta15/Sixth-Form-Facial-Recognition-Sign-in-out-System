import os
import time
from tkinter import *
from tkinter import messagebox
import cv2
from ttkbootstrap import *

from database import DatabaseHandler, RecordOperations
from detector import FaceDetector
from recogniser import FaceRecognition
from webcam import WebcamCapture

db = DatabaseHandler("localhost", "facial_recognition_manager",
                     "Faces123!", "facial_recognition_db")
record_ops = RecordOperations()


class FacialRecognitionPage:
    def __init__(self, root):
        self.root = root  # reference to the root window
        self.page = Toplevel(self.root)  # Create a Toplevel window
        self.page.geometry("1280x1240")
        self.page.title("Facial Recognition Page")
        # WIDGETS
        self.video_label = ttk.Label(self.page)
        self.video_label.place(x=10, y=10)

        self.face_id_label = ttk.Label(self.page, text="Identity:", foreground="red")
        self.face_id_label.place(x=10, y=310)

        # Instruction labels underneath webcam feed
        self.face_id_instruction_label1 = ttk.Label(self.page,
                text="Look into the webcam for facial recognition. Ensure your entire face is visible to the camera",
                font=("", 15))
        self.face_id_instruction_label1.place(x=10, y=400)

        self.face_id_instruction_label2 = ttk.Label(self.page,
                                            text="If your face is unknown, please use the alternative identification.",
                                            font=("", 15))
        self.face_id_instruction_label2.place(x=10, y=430)

        self.alt_id_label = ttk.Label(self.page, text="Alternative Identification:")
        self.alt_id_label.place(x=450, y=10)

        self.user_name_label = ttk.Label(self.page, text="Username:")
        self.user_name_label.place(x=450, y=50, width=70, height=25)
        self.user_name_entry = ttk.Entry(self.page, validate="key",
                                         validatecommand=(self.page.register(self.input_length_restrictor), "%P", 10))
        self.user_name_entry.place(x=530, y=50, width=156, height=30)

        self.password_label = ttk.Label(self.page, text="Password:")
        self.password_label.place(x=450, y=110, width=70, height=25)
        self.password_entry = ttk.Entry(self.page, show="*",
                                        validate="key",
                                        validatecommand=(self.root.register(self.input_length_restrictor), "%P", 30))
        self.password_entry.place(x=530, y=110, width=156, height=30)

        self.error_label = ttk.Label(self.page, text="", foreground="red")
        self.error_label.place(x=530, y=190, width=200, height=30)

        self.enter_button = ttk.Button(self.page, text="Enter", command=self.validate_username_password)
        self.enter_button.place(x=530, y=160, width=70, height=25)

        # Attributes for recognition algorithm
        self.face_roi = None
        self.count = 1

        self.face_recogniser = FaceRecognition()
        self.face_detector = FaceDetector()
        self.webcam = WebcamCapture()

        self.db = DatabaseHandler("localhost", "facial_recognition_manager",
                                  "Faces123!", "facial_recognition_db")
        self.known_faces = self.db.fetch_encoding_data()

        self.update()  # refreshing the page

    def update(self):
        # Increment the frame count
        self.count += 1
        ret, frame = self.webcam.read_frame()  # reading a frame (BGR), ret is a bool value indicating success status
        # Check if the frame is successfully captured
        if ret:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            frame = cv2.resize(frame, (0, 0), 5, fx=0.5, fy=0.5)  # reduce frame size for enhanced speed

            # Autocorrect the brightness of the frame using face region coordinates if detected
            grey_frame = self.face_detector.auto_correct_brightness(frame=frame, face_roi=self.face_roi)
            faces = self.face_detector.detect_faces(grey_frame)

            if len(faces) == 0:  # Check if no faces are detected
                self.identified = False
                # Update the face ID label to indicate an unknown identity
                self.face_id_label.config(text="Identity: Unknown", foreground="red")
            elif len(faces) > 0:
                # Get the largest face detected
                largest_face = faces[self.face_detector.find_largest_face(faces)]
                self.face_roi = self.face_detector.rect_to_tuple(largest_face)
                # select and process the frame every 25 frames for face recognition
                if self.count % 25 == 0:
                    aligned_face = self.face_recogniser.align_face(frame, grey_frame, face_rect=largest_face)
                    cv2.imshow("aligned face", aligned_face)  # Display the aligned face for debugging/testing

                    try:
                        # Get the face encoding for the aligned face
                        encoding = self.face_recogniser.get_encoding(aligned_face,
                                                                     self.face_detector.detect_faces(aligned_face)[0])
                        # json_encoding = json.dumps(list(encoding))  # CONVERTING THE ENCODING INTO JSON
                        # Compare the face encoding with known faces in the database
                        for identifier, known_encoding in self.known_faces.items():
                            # identifier is the index = studentID
                            if self.face_recogniser.compare_encodings(known_encoding, encoding)[0]:
                                # Match found, update face ID label with the student's name
                                match = self.db.get_student_name(identifier)
                                print(f"Match is: {match}")
                                self.face_id_label.config(text=f"Identity: {match}", foreground="green")
                                self.switch_to_sign_in_out(student_id=identifier)  # switching to sign in/out page
                                return
                            else:
                                # No match found, update face ID label as unknown
                                self.face_id_label.config(text="Identity: Unknown", foreground="red")

                    except IndexError:
                        # Handle the case where no face is found
                        self.face_id_label.config(text="Identity: Unknown", foreground="red")
                        pass

                    # Draw bounding box around the detected face
                self.face_detector.draw_bounding_box(frame, largest_face)
            else:
                # Reset the face ROI if not processing for face recognition
                self.face_roi = None

            # Convert the frame to RGB format for displaying in Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the video label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Schedule the next update after 10 milliseconds
            self.video_label.after(10, self.update)

    def validate_username_password(self):
        # getting the input from the entry boxes
        username = self.user_name_entry.get()
        password = self.password_entry.get()
        # fetching admin credentials from the database for the given username
        admin_creds = db.get_admin_credentials(username)
        # Checking if admin credentials exist and the entered password matches
        if admin_creds and admin_creds["password"] == password:
            # If admin credentials are valid, switch to the Sign In/Out page for admin
            self.switch_to_admin_dashboard(admin_credentials=admin_creds)
            return
        # Fetching student password from the database for the given username
        db_password = self.db.fetch_password(username)
        if db_password is None or password != db_password[0]:
            self.error_label.config(text="Incorrect username or password")
        else:
            self.error_label.config(text="")  # reset the error label
            self.switch_to_sign_in_out(username=username)  # Switch to the Sign In/Out page

    def input_length_restrictor(self, new_value, max_length):
        return len(new_value) <= int(max_length)

    def switch_to_sign_in_out(self, student_id=None, username=None):
        self.page.destroy()
        SignInOutPage(self.root, student_id=student_id, username=username)

    def switch_to_admin_dashboard(self, admin_credentials):
        self.page.destroy()
        AdminDashboard(self.root, admin_credentials)


# SignInOutPage class
class SignInOutPage:
    def __init__(self, root, student_id=None, username=None):
        # Initialize Toplevel window for Sign In/Out page
        self.root = root
        self.page = Toplevel(self.root)
        self.page.geometry("800x600")
        self.page.title("Sign In/Out Page")
        # WIDGETS
        # Getting the identified Students data:
        if student_id is not None:
            self.data = db.fetch_student_data(student_id=student_id)
        else:
            self.data = db.fetch_student_data(username=username)
        # making the text for the label based on data from database.
        text=f"""{self.data['first_name']} {self.data['last_name']},
        Yr{self.data['year_group']}, Form: {self.data['form']}"""
        # calculating the current state of the student - are they signed in or not
        if db.get_current_sign_in_status(self.data["student_id"]):
            self.current_state = "Signed in"
        else:
            self.current_state = "Signed out"

        # Label to display student data
        self.student_data_label = ttk.Label(self.page, text=text, font=('', 20), anchor="center",
                                            justify="center")
        self.student_data_label.place(x=10, y=10)

        self.confirm_button = ttk.Button(self.page, text="Confirm Identity", command=self.confirm_identity)
        self.confirm_button.place(x=10, y=120)

        # Label to display the current state of the student
        self.current_state_label = ttk.Label(self.page, text=f"Current state: {self.current_state}")
        self.current_state_label.place(x=10, y=40)

        # Label to go alongside switch window button
        self.not_you_label = ttk.Label(self.page, text="Not You?", font=('', 12))
        self.not_you_label.place(x=10, y=70)
        # Button to switch back to the Facial Recognition page
        self.switch_button = ttk.Button(self.page, text="Back to Identification Page",
                                        command=self.switch_to_facial_recognition)
        self.switch_button.place(x=90, y=70)

        # Label to display the current time (top right corner of the entire window)
        self.current_time = time.strftime('%H:%M:%S')  # getting the time in a string format
        current_time_str = f"Current time: {self.current_time}"
        self.current_time_label = ttk.Label(self.page, text=current_time_str, font=('', 14))
        self.current_time_label.pack(side="top", padx=10, pady=10, anchor="e")
        student_on_time = False

    def confirm_identity(self):
        # Method to confirm identity before proceeding with sign-in/sign-out
        self.confirm_button.destroy()
        if self.current_state == "Signed in" and self.current_time < "15:05:00":
            # Label and reason buttons for "Early SIGN OUT reasons"
            self.early_sign_out_label = ttk.Label(self.page, text="Early SIGN OUT reasons")
            self.early_sign_out_label.place(x=10, y=130, anchor="nw")
            student_on_time = False
            ttk.Button(self.page, text="Appointment", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_out(self.data["student_id"],
                                                                early_reason="Appointment")).place(
                x=10, y=200, anchor="nw")
            ttk.Button(self.page, text="Emergency", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_out(self.data["student_id"],
                                                                early_reason="Emergency")).place(
                x=150, y=200, anchor="nw")
            ttk.Button(self.page, text="Extra-curricular", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_out(self.data["student_id"],
                                                                early_reason="Extra-curricular")).place(x=290, y=200,
                                                                                                        anchor="nw")
            ttk.Button(self.page, text="Illness", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_out(self.data["student_id"], early_reason="Illness")).place(
                x=10,
                y=240,
                anchor="nw")
            ttk.Button(self.page, text="Lunch (offsite)", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_out(self.data["student_id"],
                                                                early_reason="Lunch (offsite)")).place(x=150, y=240,
                                                                                                       anchor="nw")
            ttk.Button(self.page, text="Other", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_out(self.data["student_id"], early_reason="Other")).place(
                x=290,
                y=240,
                anchor="nw")

        elif self.current_state == "Signed out" and self.current_time > "09:05:00":
            # Label and reason buttons for "Late SIGN in reasons"
            self.late_sign_in_label = ttk.Label(self.page, text="Late SIGN IN reasons")
            self.late_sign_in_label.place(x=10, y=160, anchor="nw")
            student_on_time = False
            ttk.Button(self.page, text="Traffic", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_in(self.data["student_id"], late_reason="Traffic")).place(
                x=10,
                y=200,
                anchor="nw")
            ttk.Button(self.page, text="Illness", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_in(self.data["student_id"], late_reason="Illness")).place(
                x=150,
                y=200,
                anchor="nw")
            ttk.Button(self.page, text="Appointment", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_in(self.data["student_id"],
                                                               late_reason="Appointment")).place(
                x=290, y=200, anchor="nw")
            ttk.Button(self.page, text="Emergency", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_in(self.data["student_id"], late_reason="Emergency")).place(
                x=10,
                y=240,
                anchor="nw")
            ttk.Button(self.page, text="Lunch return", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_in(self.data["student_id"],
                                                               late_reason="Lunch return")).place(
                x=150, y=240, anchor="nw")
            ttk.Button(self.page, text="Other", width=20, padding=5,
                       command=lambda: self.trigger_db_sign_in(self.data["student_id"], late_reason="Other")).place(
                x=290,
                y=240,
                anchor="nw")
        elif self.current_state == "Signed in":
            student_on_time = True  # they are signing out at the correct time of day
            self.trigger_db_sign_out(self.data["student_id"])
        else:
            student_on_time = True  # they are signing out at the correct time of day
            self.trigger_db_sign_in(self.data["student_id"])

    def trigger_db_sign_in(self, student_id, early_reason=None, late_reason=None):
        # sign student in with late reason if appropriate
        db.sign_in_out_student(student_id, is_sign_in=True, late_reason=late_reason)
        self.show_confirmation_message("Sign In")
        self.switch_to_facial_recognition()
        self.page.destroy()

    def trigger_db_sign_out(self, student_id, early_reason=None, late_reason=None):
        # sign student out with early reason if appropriate
        db.sign_in_out_student(student_id, is_sign_in=False, early_reason=early_reason)
        self.show_confirmation_message("Sign Out")
        self.switch_to_facial_recognition()
        self.page.destroy()

    def show_confirmation_message(self, action):
        message = f"{action} successful!"
        messagebox.showinfo("Confirmation", message)

    def switch_to_facial_recognition(self):
        # Method to switch back to the Facial Recognition page
        print("switching facial recognition page")
        self.page.destroy()
        FacialRecognitionPage(self.root)


class AdminDashboard:
    def __init__(self, root, admin_credentials):
        # Initialize Toplevel window for Sign In/Out page
        self.root = root
        self.credentials = admin_credentials
        self.page = Toplevel(self.root)
        self.page.geometry("1920x1080")
        self.page.title("Admin Dashboard")
        # Label to display admin's first name and last name
        admin_name_label = ttk.Label(self.page,
                                     text=f"Welcome, {admin_credentials['first_name']} {admin_credentials['last_name']}"
                                     , font=('', 15))
        admin_name_label.grid(row=0, column=1, pady=10)

        # Button to sign out of the admin dashboard
        sign_out_button = ttk.Button(self.page, text="Sign Out", command=self.sign_out)
        sign_out_button.grid(row=0, column=2, pady=10, padx=10, sticky="e")

        # Label for number of students onsite
        student_count_data = db.get_students_onsite_offsite_count()
        self.onsite_label = ttk.Label(self.page, text=f"No. students ONSITE: {student_count_data[0]}", font=('', 15))
        self.onsite_label.grid(row=1, column=0, pady=10, padx=10, sticky="w")
        # Label for number of students absent
        self.absent_label = ttk.Label(self.page, text=f"No. students ABSENT:{student_count_data[1]}", font=('', 15))
        self.absent_label.grid(row=2, column=0, pady=10, padx=10, sticky="w")

        # Button to refresh metrics
        refresh_button = ttk.Button(self.page, text="Refresh", command=self.refresh_metrics)
        refresh_button.grid(row=0, column=3, pady=10, padx=10, sticky="e")

        # Button to create a new student record
        create_record_button = ttk.Button(self.page, text="Create New Student Record",
                                          command=self.switch_to_new_record_page)
        create_record_button.grid(row=3, column=0, pady=10, padx=10, sticky="w")

        # TREEVIEW FOR LATE SIGN IN REASONS
        self.late_table_label = ttk.Label(self.page, text="Late Sign-In Notifications")
        self.late_table_label.grid(row=1, column=2)
        self.late_sign_in_tree = ttk.Treeview(self.page, columns=(
            'FirstName', 'LastName', 'YearGroup', 'Form', 'Time', 'LateSignInReason'), show='headings', height=10)

        self.late_sign_in_tree.heading('FirstName', text='First Name')
        self.late_sign_in_tree.heading('LastName', text='Last Name')
        self.late_sign_in_tree.heading('YearGroup', text='Year Group')
        self.late_sign_in_tree.heading('Form', text='Form')
        self.late_sign_in_tree.heading('Time', text='Late Arrival Time')
        self.late_sign_in_tree.heading('LateSignInReason', text='Late Sign-In Reason', )
        self.late_sign_in_tree.grid(row=2, column=2, rowspan=3, columnspan=3, pady=10, padx=10, sticky="nsew")

        self.late_sign_in_tree.column('FirstName', anchor=CENTER, width=110)
        self.late_sign_in_tree.column('LastName', anchor=CENTER, width=110)
        self.late_sign_in_tree.column('YearGroup', anchor=CENTER, width=110)
        self.late_sign_in_tree.column('Form', anchor=CENTER, width=110)
        self.late_sign_in_tree.column('Time', anchor=CENTER, width=110)
        self.late_sign_in_tree.column('LateSignInReason', anchor=CENTER, width=110)

        self.page.columnconfigure(1, weight=1)
        self.page.columnconfigure(2, weight=1)
        self.page.columnconfigure(3, weight=1)
        self.page.columnconfigure(4, weight=1)
        self.page.columnconfigure(5, weight=1)
        self.page.columnconfigure(6, weight=1)

        # Call the method to populate notifications
        self.populate_late_sign_in_table(data=db.get_late_students())

        # TREEVIEW FOR EARLY SIGN OUT RECORDS
        self.early_table_label = ttk.Label(self.page, text="Early Sign-Out Notifications")
        self.early_table_label.grid(row=6, column=2)
        self.early_sign_out_tree = ttk.Treeview(self.page, columns=(
            'FirstName', 'LastName', 'YearGroup', 'Form', 'Time', 'EarlySignOutReason'), show='headings', height=10)

        self.early_sign_out_tree.heading('FirstName', text='First Name')
        self.early_sign_out_tree.heading('LastName', text='Last Name')
        self.early_sign_out_tree.heading('YearGroup', text='Year Group')
        self.early_sign_out_tree.heading('Form', text='Form')
        self.early_sign_out_tree.heading('Time', text='Early Sign Out Time')
        self.early_sign_out_tree.heading('EarlySignOutReason', text='Early Sign-Out Reason', )
        self.early_sign_out_tree.grid(row=7, column=2, rowspan=3, columnspan=3, pady=10, padx=10, sticky="nsew")

        self.early_sign_out_tree.column('FirstName', anchor=CENTER, width=110)
        self.early_sign_out_tree.column('LastName', anchor=CENTER, width=110)
        self.early_sign_out_tree.column('YearGroup', anchor=CENTER, width=110)
        self.early_sign_out_tree.column('Form', anchor=CENTER, width=110)
        self.early_sign_out_tree.column('Time', anchor=CENTER, width=110)
        self.early_sign_out_tree.column('EarlySignOutReason', anchor=CENTER, width=110)
        # Call the method to populate notifications
        self.populate_early_sign_out_table(data=db.get_early_students())

        self.generate_registers_button = ttk.Button(self.page, text="Generate Evacuation Registers",
                                                    command=self.generate_registers)
        self.generate_registers_button.grid(row=4, column=0, pady=10, padx=10, sticky="w")

    def generate_registers(self):
        # Define column names
        print("pdfs generated")
        columns = ['First Name', 'Last Name', 'Sign In Time', 'Sign out time']
        # Output PDF file
        for form_group in range(1, 9):
            # Call the function with data, columns, and filename for year 12s
            data = db.get_evacuation_register_data(12, form_group)
            pdf_filename = f'Register_year12_form{form_group}.pdf'
            db.generate_pdf(pdf_filename, data, columns, 12, form_group)
        for form_group in range(1, 9):
            # Call the function with data, columns, and filename for year 13s
            pdf_filename = f'Register_year13_form{form_group}.pdf'
            data = db.get_evacuation_register_data(13, form_group)
            db.generate_pdf(pdf_filename, data, columns, 13, form_group)

        messagebox.showinfo("Confirmation", "Reports successfully generated")

    def populate_late_sign_in_table(self, data):
        # Clear existing items in the Treeview
        self.late_sign_in_tree.delete(*self.late_sign_in_tree.get_children())

        # Populate the Treeview with data
        for student_record in data:
            values = (
                student_record.get("FirstName", ""),
                student_record.get("LastName", ""),
                student_record.get("YearGroup", ""),
                student_record.get("Form", ""),
                student_record.get("SignInTime", ""),
                student_record.get("LateSignInReason", "")
            )
            self.late_sign_in_tree.insert("", "end", values=values)

    def populate_early_sign_out_table(self, data):
        # Clear existing items in the Treeview
        self.early_sign_out_tree.delete(*self.early_sign_out_tree.get_children())

        # Populate the Treeview with data
        for student_record in data:
            values = (
                student_record.get("FirstName", ""),
                student_record.get("LastName", ""),
                student_record.get("YearGroup", ""),
                student_record.get("Form", ""),
                student_record.get("SignOutTime", ""),
                student_record.get("EarlySignOutReason", "")
            )
            self.early_sign_out_tree.insert("", "end", values=values)

    # Refresh function to update the notifications
    def refresh_notifications(self):
        """Refreshes the notifications in the Treeview."""
        # Clear existing data
        for item in self.late_sign_in_tree.get_children():
            self.late_sign_in_tree.delete(item)
        # Populate with updated data
        self.populate_late_sign_in_table(db.get_late_students())

        for item in self.early_sign_out_tree.get_children():
            self.early_sign_out_tree.delete(item)
        # Populate with updated data
        self.populate_early_sign_out_table(db.get_early_students())

    def sign_out(self):
        # Method to sign out of the admin dashboard and return to the recognition screen
        self.page.destroy()
        FacialRecognitionPage(self.root)

    def refresh_metrics(self):
        """Refreshes the metrics on the admin dashboard."""
        results = db.get_students_onsite_offsite_count()  # update tables with most recent notifications.
        # Update labels with refreshed metrics
        self.onsite_label.config(text=f"No. students ONSITE:{results[0]}")
        self.absent_label.config(text=f"No. students ABSENT:{results[1]}")
        self.refresh_notifications()

    def switch_to_new_record_page(self):
        print("switching to new record page")
        NewStudentRecordPage(self.root)


class NewStudentRecordPage:
    def __init__(self, root):
        self.root = root
        self.add_window = Toplevel(root)
        self.add_window.title("Add New Student Record")

        self.username = None  # when the username is generated when a new record is added, this variable is updated

        # Entry Fields and Labels
        self.first_name_label = Label(self.add_window, text="First Name:")
        self.first_name_entry = Entry(self.add_window, width=30)
        self.last_name_label = Label(self.add_window, text="Last Name:")
        self.last_name_entry = Entry(self.add_window, width=30)
        self.password_label = Label(self.add_window, text="Password:")

        self.password_entry = Entry(self.add_window, show="*", width=30, validate="key",
                                    validatecommand=(self.root.register(self.input_length_restrictor), "%P", 30))

        self.confirm_password_label = Label(self.add_window, text="Confirm Password:")

        self.confirm_password_entry = Entry(self.add_window, show="*", width=30, validate="key",
                                            validatecommand=(
                                                self.root.register(self.input_length_restrictor), "%P", 30))
        self.year_group_label = Label(self.add_window, text="Year Group:")
        self.year_group_entry = Entry(self.add_window, width=30)
        self.form_group_label = Label(self.add_window, text="Form Group:")
        self.form_group_entry = Entry(self.add_window, width=30)
        self.entry_year_label = Label(self.add_window, text="Entry Year:")
        self.entry_year_entry = Entry(self.add_window, width=30)
        self.image_path_label = Label(self.add_window, text="Image Path:")
        self.image_path_entry = Entry(self.add_window, width=30)
        self.generate_username_label = Label(self.add_window, text="us:")
        self.generate_username_button = Button(self.add_window, text="Generate Username",
                                               command=self.generate_username)

        # Button to add record to db
        # once the add record button is pressed, inputs of the form will first be validated.
        self.add_button = Button(self.add_window, text="Add Student Record", command=self.validate_inputs)

        self.error_label = Label(self.add_window, text="", foreground="red")
        # Packing Entry Fields and Buttons with Labels
        self.first_name_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.first_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.last_name_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.last_name_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.password_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.password_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.confirm_password_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.confirm_password_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.year_group_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.year_group_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.form_group_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.form_group_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.entry_year_label.grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.entry_year_entry.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.image_path_label.grid(row=7, column=0, padx=5, pady=5, sticky="e")
        self.image_path_entry.grid(row=7, column=1, padx=5, pady=5, sticky="w")
        self.generate_username_label.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        self.generate_username_button.grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.add_button.grid(row=9, column=0, columnspan=2, pady=20)
        self.error_label.grid(row=10, column=0, columnspan=2, pady=20)

    def input_length_restrictor(self, new_value, max_length):
        return len(new_value) <= int(max_length)

    def validate_inputs(self):
        # Get values from entry fields
        first_name = self.first_name_entry.get()
        last_name = self.last_name_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_password_entry.get()
        year_group = self.year_group_entry.get()
        form_group = self.form_group_entry.get()
        entry_year = self.entry_year_entry.get()
        image_path = self.image_path_entry.get()

        # Check if any field is empty
        if not all([first_name, last_name, password, confirm_password, year_group, form_group, entry_year, image_path]):
            self.error_label.config(text="All fields must be filled.")
            return False

        # Check if year_group, form_group, and entry_year are integers
        try:
            year_group = int(year_group)
            form_group = int(form_group)
            entry_year = int(entry_year)
        except ValueError:
            self.error_label.config(text="Year group, form group, and entry year must all be integers.")
            return False

        # Check if entry_year in correct range.
        if entry_year < 2000 or entry_year > 2099:
            self.error_label.config(text="Invalid year: Entry year must be between 2000-2100.")
            return False
        # Check that a username has been generated
        if not self.username:  # if the value of username (in __init__) is None, this is True
            self.error_label.config(text="Generate a username")
            return False

        # Check if passwords match
        if password != confirm_password:
            self.error_label.config(text="Passwords do not match.")
            return False

        # Check if the image path exists and is accessible
        if not os.path.exists(image_path) or not os.access(image_path, os.R_OK):
            self.error_label.config(text="Invalid image path. Please provide a valid path.")
            return False

        try:
            encoding = FaceRecognition().train_new_face(image_path)
        except:  # If there is an error this will handle it - means the image is not adequate
            self.error_label.config(text="Face cannot be detected/encoded in this image.")
            return False
        print("All tests passed")
        self.add_student_record(username=self.username,
                                password=password,
                                firstname=first_name,
                                lastname=last_name,
                                yeargroup=year_group,
                                form=form_group,
                                image_path=image_path
                                )

    def generate_username(self):
        lastname = self.last_name_entry.get()
        entry_year = self.entry_year_entry.get()
        if not lastname:
            self.username = record_ops.generate_username(entry_year, self.first_name_entry.get())
        else:
            self.username = record_ops.generate_username(entry_year, self.last_name_entry.get())
        self.generate_username_label.config(text=f"{self.username}")

    def add_student_record(self, username, password, firstname, lastname, yeargroup, form, image_path):
        try:
            db.add_new_record(username, password, firstname, lastname, yeargroup, form, image_path)
            # Close the window after successful record addition
            self.add_window.destroy()
            # Show a confirmation messagebox
            messagebox.showinfo("Success", "Student record added successfully!")
        except Exception as e:
            # If there is an error during record addition, show an error messagebox
            messagebox.showerror("Error", f"Error adding student record: {str(e)}")


# MainApplication class
class MainApplication:
    def __init__(self):
        # Initialize the main Tkinter window
        self.root = Tk()
        self.root.geometry("10x10")
        self.root.title("Main Application")
        self.root.iconify()

        # Initialize the Facial Recognition page
        FacialRecognitionPage(self.root)
        # SignInOutPage(self.root, 1)
        AdminDashboard(self.root, {'first_name': 'Bob', 'last_name': 'Smith', 'password': 'admin123!'})
        # NewStudentRecordPage(self.root)

    # Method to run the Tkinter main loop
    def run(self):
        self.root.mainloop()


# Main block to run the application
if __name__ == "__main__":
    app = MainApplication()
    app.run()
