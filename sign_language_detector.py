import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants and parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
          "U", "V", "W", "X", "Y", "Z", "I LOVE YOU", "THANK YOU", "One", "Two", "Three", "Four", "Five", "Six",
          "Seven", "Eight", "Nine"]

while True:
    # Read frame from the video capture
    success, img = cap.read()
    imgOutput = img.copy()

    # Detect hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # x, y, width, and height

        # Create a white canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)

        # Crop the hand region from the frame
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize and center the cropped image
        if h > w:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Get prediction and index from the classifier
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display prediction and bounding box
        cv2.putText(imgOutput, labels[index], (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 255, 255), 4)

        # Display cropped and resized images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display output image
    cv2.imshow("Image", imgOutput)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
