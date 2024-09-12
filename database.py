import datetime
import json
import mysql.connector
import numpy as np
import pandas as pd
from mysql.connector import Error
from recogniser import FaceRecognition

# imports for generating registers using reportlabs:
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.platypus.para import Paragraph


class DatabaseHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def create_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if connection.is_connected():
                # print("Connected to MySQL database")
                return connection
        except Error as e:
            print(f"Error: {e}")
            return None

    def close_connection(self, connection):
        if connection.is_connected():
            connection.close()
            # print("Connection closed")

    def insert_student_data(self, id, data):
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                # Using parameterised query to prevent SQL injection
                sql = "INSERT INTO facialencodings (StudentID, FacialEncoding) VALUES (%s, %s)"
                values = (id, json.dumps(data["facial_encoding"]))
                cursor.execute(sql, values)
                connection.commit()
                print("Record inserted successfully")
            except Error as e:
                print(f"Error: {e}")
                return None

            finally:
                if connection.is_connected():
                    if cursor: cursor.close()
                    self.close_connection(connection)

    def get_student_name(self, student_id):
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                query = "SELECT FirstName, LastName FROM Students WHERE StudentID = %s"
                cursor.execute(query, (student_id,))  # passing in the data parameter when executing query
                result = cursor.fetchone()  # fetching the single relevant row
                if result:
                    # Assuming the query returns (FirstName, LastName)
                    first_name, last_name = result
                    full_name = f"{first_name} {last_name}" if last_name else first_name  # case if theres no last name
                    return full_name
                else:
                    return None
            except Error as e:
                print(f"Error: {e}")
                return None
            finally:
                if connection:
                    if cursor: cursor.close()
                    self.close_connection(connection)

    def fetch_password(self, username):

        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                query = "SELECT Password FROM Students WHERE Username = %s"
                cursor.execute(query, (username,))  # passing in the data parameter when executing query
                result = cursor.fetchone()  # fetching the single relevant row
                if result:
                    return result
                else:
                    return None
            except Error as e:
                print(f"Error: {e}")
                return None
            finally:
                if connection:
                    if cursor: cursor.close()
                    self.close_connection(connection)

    def fetch_encoding_data(self):
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                query = "SELECT StudentID, FacialEncoding FROM facialencodings"  # the sql query
                cursor.execute(query)
                # Fetching ALL rows
                facial_encodings_data = cursor.fetchall()  # each row is a tuple format: (id, encoding)
                # Process the data
                data = {}
                for id, encoding in facial_encodings_data:
                    data[id] = np.array(json.loads(encoding))  # converting json back into np array for recognition
                return data

            except Error as e:
                print(f"Error: {e}")
                return None

            finally:
                # closing connections
                if connection.is_connected():
                    if cursor: cursor.close()
                    self.close_connection(connection)

    def fetch_student_data(self, student_id=None, username=None):  # data for the sign in/out  page for GUI
        connection = self.create_connection()
        if not connection:
            return None
        cursor = None
        try:
            cursor = connection.cursor()
            # parameterised query depending on which data is given to us
            if student_id:
                query = "SELECT StudentID, FirstName, LastName, YearGroup, Form FROM Students WHERE StudentID = %s"
                cursor.execute(query, (student_id,))
            elif username:
                query = "SELECT StudentID, FirstName, LastName, YearGroup, Form FROM Students WHERE Username = %s"
                cursor.execute(query, (username,))
            result = cursor.fetchone()
            if result:
                student_id, first_name, last_name, year_group, form = result
                return {
                    "student_id": student_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "year_group": year_group,
                    "form": form
                }  # data in a dictionary format
            else:
                return None
        # handling the connection once complete
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            if connection.is_connected():
                if cursor:
                    cursor.close()
                self.close_connection(connection)

    def sign_in_out_student(self, student_id, is_sign_in, late_reason=None, early_reason=None):
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                if is_sign_in and self.is_student_signed_in_today(student_id):
                    print("Error: Student is already signed in for today. Please sign out first.")
                    return
                if not is_sign_in and not self.is_student_signed_in_today(student_id):
                    print("Error: Student is not signed in for today. Please sign in first.")
                    return

                cursor = connection.cursor()
                # Get the current date and time
                current_date = datetime.date.today()
                current_time = datetime.datetime.now().time()
                # Determine whether it's a sign-in or sign-out
                if is_sign_in:
                    # create a new record
                    query1 = ("INSERT INTO AttendanceLog (StudentID, Date, SignInTime, LateSignInReason)"
                              "VALUES (%s, %s, %s, %s)")
                    values = (student_id, current_date, current_time, late_reason)
                else:
                    # if signing out we need to find the record where the user has signed in and update that record

                    query = ("UPDATE AttendanceLog AS a JOIN (SELECT StudentID, Date, MAX(LogID) AS MaxLogID "
                             "FROM AttendanceLog WHERE StudentID = %s AND Date = %s GROUP BY StudentID, Date) "
                             "AS max_log ON a.StudentID = max_log.StudentID AND a.Date = max_log.Date "
                             "AND a.LogID = max_log.MaxLogID SET a.SignOutTime = %s, a.EarlySignOutReason = %s;")
                    values = (student_id, current_date, current_time, early_reason)

                cursor.execute(query, values)
                connection.commit()
                print("Sign-in/out record updated successfully")

            except Error as e:
                print(f"Error: {e}")
                return None

            finally:
                if connection.is_connected():
                    if cursor:
                        cursor.close()
                    self.close_connection(connection)

    def get_current_sign_in_status(self, student_id):
        """get the sign in status of a student - returns true if signed in and false if signed out"""
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                current_date = datetime.date.today()
                query = ("SELECT StudentID FROM AttendanceLog WHERE StudentID = %s AND Date = %s"
                         " AND SignInTime IS NOT NULL AND SignOutTime IS NULL")
                values = (student_id, current_date)
                cursor.execute(query, values)

                if cursor.fetchone() is None:  # if tuple is empty
                    return False  # it means that the student has not signed in today,
                    # hence is currently signed out

                return True  # if value returned, this means that the user is currently signed in but not signed out
                # - hence they want to sign out

            except Error as e:
                print(f"Error: {e}")
                return False
            finally:
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def is_student_signed_in_today(self, student_id):
        """Finds the latest sign-in for a particular student today -
        used for auto-signing out student/preventing duplicate signin/out"""
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                current_date = datetime.date.today()
                query = ("SELECT * FROM AttendanceLog WHERE StudentID = %s AND Date = %s "
                         "AND SignInTime IS NOT NULL AND SignOutTime IS NULL")
                values = (student_id, current_date)
                cursor.execute(query, values)
                return cursor.fetchone() is not None

            except Error as e:
                print(f"Error: {e}")
                return False
            finally:
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def auto_sign_out_students(self):
        # at a certain time each day this function will be triggered to automatically sign out any students
        # who were signed in and have not signed out of the site.
        # assume that they have left.
        pass

    def get_admin_credentials(self, admin_username):
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                query = "SELECT FirstName, LastName, Password FROM Admins WHERE Username = %s"
                cursor.execute(query, (admin_username,))  # passing in the data parameter when executing query
                result = cursor.fetchone()  # fetching the single relevant row
                if result is not None:
                    first_name, last_name, password = result
                    return {
                        "first_name": first_name,
                        "last_name": last_name,
                        "password": password
                    }  # data in a dictionary format
                else:
                    return None
            except Error as e:
                print(f"Error: {e}")
                return None
            finally:
                if connection:
                    if cursor: cursor.close()
                    self.close_connection(connection)

    def get_total_students_count(self):
        """Fetches and returns the total count of students in the database."""
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()  # Create a cursor for executing SQL queries.
                # SQL query to count the total number of students in the Students table.
                query = "SELECT COUNT(*) FROM Students"
                cursor.execute(query)
                result = cursor.fetchone()  # Fetch the result of the query.
                return result[0] if result else 0  # Return the count if result is not None, otherwise return 0.
            except Error as e:
                print(f"Error: {e}")
                return 0  # Return 0 in case of an error during execution.
            finally:
                if connection:
                    if cursor: cursor.close()  # Close the cursor.
                    self.close_connection(connection)  # Close the database connection.

    def get_students_onsite_offsite_count(self):
        """Counts the number of students who are signed in.
        Then returns the number of students who are present and absent based on this count."""
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()  # Create a cursor for executing SQL queries.
                current_date = datetime.date.today()
                # SQL query to count the number of students signed in but not signed out for the current date.
                query = ("SELECT COUNT(*) FROM AttendanceLog WHERE Date = %s AND SignInTime"
                         " IS NOT NULL AND SignOutTime IS NULL")
                values = (current_date,)
                cursor.execute(query, values)
                result = cursor.fetchone()  # Fetch the result of the query.
                # print(result)
                if result is not None:
                    # Calculate the count of onsite students and offsite students.
                    onsite_count = result[0]
                    offsite_count = self.get_total_students_count() - onsite_count

                    return onsite_count, offsite_count  # Return the counts as a tuple.
            except Error as e:
                print(f"Error: {e}")
                return False  # Return False in case of an error during execution.
            finally:
                if connection:
                    cursor.close()  # Close the cursor.
                    self.close_connection(connection)  # Close the database connection.

    def get_late_students(self):
        """Retrieve records of students who arrived late."""
        # Establish a connection to the database
        connection = self.create_connection()
        todays_date = datetime.date.today()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor(dictionary=True)
                # SQL query to retrieve details of students who arrived late
                query = """
                SELECT 
                    Students.FirstName,
                    Students.LastName,
                    Students.Username,
                    Students.YearGroup,
                    Students.Form,
                    AttendanceLog.Date,
                    AttendanceLog.SignInTime,
                    AttendanceLog.LateSignInReason
                FROM
                    Students
                INNER JOIN
                    AttendanceLog ON Students.StudentID = AttendanceLog.StudentID
                WHERE
                    AttendanceLog.Date = %s AND
                    AttendanceLog.SignInTime IS NOT NULL

                    AND AttendanceLog.LateSignInReason IS NOT NULL;
                """

                cursor.execute(query, (todays_date,))
                late_students = cursor.fetchall()  # Fetch all the late students' records
                # print(late_students)
                return late_students
            except Error as e:
                print(f"Error: {e}")
                return []
            finally:
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def get_early_students(self):
        """Retrieve records of students who arrived late."""
        # Establish a connection to the database
        todays_date = datetime.date.today()
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor(dictionary=True)
                # SQL query to retrieve details of students who arrived late
                query = """
                SELECT 
                    Students.FirstName,
                    Students.LastName,
                    Students.Username,
                    Students.YearGroup,
                    Students.Form,
                    AttendanceLog.Date,
                    AttendanceLog.SignOutTime,
                    AttendanceLog.EarlySignOutReason
                FROM
                    Students
                INNER JOIN
                    AttendanceLog ON Students.StudentID = AttendanceLog.StudentID
                WHERE
                    AttendanceLog.Date = %s AND
                    AttendanceLog.SignInTime IS NOT NULL
                    AND AttendanceLog.SignOutTime IS NOT NULL
                    AND AttendanceLog.EarlySignOutReason IS NOT NULL;
                """

                cursor.execute(query, (todays_date,))
                early_students = cursor.fetchall()  # Fetch all the early sign-out students' records
                # print(early_students)
                return early_students

            except Error as e:
                print(f"Error: {e}")
                return []
            finally:
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def get_all_usernames(self):
        """Retrieve all usernames from the student table."""
        connection = self.create_connection()
        if connection:
            cursor = None
            try:
                cursor = connection.cursor()
                # SQL query to retrieve all usernames
                query = "SELECT Username FROM Students;"
                # Execute the SQL query
                cursor.execute(query)
                # Fetch all usernames
                usernames = cursor.fetchall()
                # Return the result
                return usernames
            except Error as e:
                # Handle any errors that might occur during the execution
                print(f"Error: {e}")
                return []
            finally:
                # Close the cursor and the database connection
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def add_new_record(self, username, password, firstname, lastname, yeargroup, form, image_path):
        # Step 1: Add a new student to the Students table
        connection = self.create_connection()
        if connection:
            cursor = connection.cursor()
            try:
                query_add_student = """
                INSERT INTO Students (Username, Password, FirstName, LastName, YearGroup, Form)
                VALUES (%s, %s, %s, %s, %s, %s);
                """
                values_student = (username, password, firstname, lastname, yeargroup, form)
                cursor.execute(query_add_student, values_student)
                connection.commit()
                print("Student data successfully added")
                # Retrieve the auto-generated student ID
                student_id = cursor.lastrowid
                if student_id:
                    # Step 2: Capture the student's facial encoding
                    try:
                        encoding = recogniser.train_new_face(image_path=image_path)
                        # Step 3: Add the facial encoding to the FacialEncodings table
                        query_add_encoding = """
                        INSERT INTO FacialEncodings (StudentID, FacialEncoding)
                        VALUES (%s, %s);
                        """
                        values_encoding = (student_id, json.dumps(list(encoding)))
                        cursor.execute(query_add_encoding, values_encoding)
                        connection.commit()
                        print("Facial encoding successfully added")
                    except Exception as e:
                        print(f"Error capturing facial encoding: {e}")
            except Exception as e:
                print(f"Error adding new student: {e}")
            finally:
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def get_evacuation_register_data(self, year, form):
        connection = self.create_connection()  # connection to database
        if connection:
            cursor = None
            try:
                # parameterised SQL query to extract relevant data
                # this query finds the latest sign in time for each student for todays date.
                # if the latest sign in time is less than the latest sign out time,
                # then the user has signed in and out on that day
                # however if the latest sign out time is less than the sign in time,
                # then the user has signed in but not out yet (they are on sight)
                # this is necessary for students who signed in/out multiple times in a day.
                query = """SELECT
    Students.FirstName,
    Students.LastName,
    MAX(AttendanceLog.SignInTime) AS SignInTime,
    CASE
        WHEN MAX(AttendanceLog.SignOutTime) > MAX(AttendanceLog.SignInTime)
            THEN MAX(AttendanceLog.SignOutTime)
        ELSE NULL
    END AS SignOutTime
FROM Students
LEFT JOIN (
    SELECT 
        StudentID,
        MAX(CASE WHEN Date = CURDATE() THEN SignInTime ELSE NULL END) AS SignInTime,
        MAX(CASE WHEN Date = CURDATE() THEN SignOutTime ELSE NULL END) AS SignOutTime
    FROM 
        AttendanceLog
    GROUP BY 
        StudentID
) AS AttendanceLog ON Students.StudentID = AttendanceLog.StudentID
WHERE Students.Form = %s AND Students.YearGroup = %s 
GROUP BY Students.FirstName, Students.LastName
ORDER BY Students.LastName;
"""
                # setting up the cursor
                cursor = connection.cursor()  # so each record is returned as a 2d tuple
                cursor.execute(query, (form, year,))  # passing in parameters
                data = cursor.fetchall()  # getting the data

                # Convert timedelta objects to strings with time part only
                data_formatted = [(first_name, last_name, str(sign_in).split()[-1] if pd.notnull(sign_in) else '',
                                   str(sign_out).split()[-1] if pd.notnull(sign_out) else '')
                                  for first_name, last_name, sign_in, sign_out in data]
                # print(data_formatted)
                return data_formatted
            except Exception as e:
                print(f"Error retrieving data: {e}")
            finally:
                # Close the cursor and the database connection
                if connection:
                    cursor.close()
                    self.close_connection(connection)

    def generate_pdf(self, pdf_filename, data, columns, year_group, form_number):
        # Create a Pandas DataFrame
        df = pd.DataFrame(data, columns=columns)
        df.insert(len(df.columns), 'Mark Attendance', '')  # adding the extra column at the end

        # Convert the DataFrame to a list of lists for reportlab
        pdf_table = [df.columns.tolist()] + df.values.tolist()

        # Create a PDF document
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        elements = []

        # Add title to the document
        title_text = f"Evacuation Register - Year {year_group}, Form {form_number}"
        elements.append(Paragraph(title_text, getSampleStyleSheet()['Title']))

        # Create a table from the DataFrame
        col_widths = [130, 130, 100, 100, 100]
        table = Table(pdf_table, colWidths=col_widths)

        # Style the table
        style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)])

        table.setStyle(style)
        elements.append(table)

        # Build the PDF document
        doc.build(elements)


class RecordOperations:
    @staticmethod
    def generate_username(entryYear, last_name):
        existing_usernames = [username[0] for username in
                              db.get_all_usernames()]  # retrieves all usernames from the database
        print(existing_usernames)
        if len(last_name) < 5:
            base_username = f"{str(entryYear)[-2:]}{last_name.upper()}"  # if the last name < 5 characters
        else:
            base_username = f"{str(entryYear)[-2:]}{last_name.upper()[:5]}"

        if base_username not in existing_usernames:
            return base_username

        # If the base username already exists, find a unique one by adding a number
        index = 2
        while f"{base_username}{index}" in existing_usernames:
            index += 1
        return f"{base_username}{index}"


recogniser = FaceRecognition()
test_data = []

ro = RecordOperations()
db = DatabaseHandler("localhost", "facial_recognition_manager",
                     "Faces123!", "facial_recognition_db")

db.get_evacuation_register_data(12, 2)
