import dlib
import cv2

detector = dlib.get_frontal_face_detector()  # Load the pre-trained facial detection model from dlib
cap = cv2.VideoCapture(0)  # Open the webcam (0 is the default camera)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()  # ret is boolean indicating if the frame was successfully read or not
    if ret:  # processing only if the frame is successfully read
        frame = cv2.flip(frame, 1)  # flipping the frame in the y-axis to make the feed mirror-like.
        # Convert the frame to grayscale (required by dlib)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)  # Detect faces in the grayscale frame using the dlib detector

        # Draw rectangles around the detected faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()  # getting coordinates of the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Facial Detection', frame)
    # Break the loop if the 'q' key is pressed ending the camera feed - for testing purposes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture
cap.release()
cv2.destroyAllWindows()
