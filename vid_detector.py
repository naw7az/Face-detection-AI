import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# use 0(to capture video from webcam) or the absolute path of an actual video 
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to Grayscale
    grayscale_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_vid) 

    # Draw rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    # To show video
    cv2.imshow('Vid', frame)

    # wait for 1ms and press a key automatically and goes to next frame
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key==81 or key==113: # (ASCII char of Q and q)
        break

# Release the VideoCapture object
webcam.release()

print("Code Completed .....")