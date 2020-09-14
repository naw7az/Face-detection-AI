import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv github(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# this code is needed sometimes to tell window to show the image
# cv2.namedWindow('Images')

# Choose an image to detect faces in
img = cv2.imread('C:/Users/nawaz/OneDrive/Desktop/Face Detection AI/images/bp.jpg')

# Must convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
# print(face_coordinates) # [top-left co-ordinate(x,y), width, height]


# Draw rectangle around the faces
# cv2.rectangle(img, top-left coordinate(x,y), (x+w, y+h), BGR(reverse of RGB), thickness of square)
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 4)

# to show color image
cv2.imshow('Images', img)


# to show grayscale image
# cv2.imshow('Images', grayscale_img)

# to pause until a key is pressed
cv2.waitKey()

print("Code Completed .....")