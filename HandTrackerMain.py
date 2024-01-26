import cv2
import mediapipe as mp
import time
from HandTrackingMin import handDetector


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
#1 for webcam

detector = handDetector()
while True:  
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False) 
    if len(lmList) != 0 :
        print(lmList[0])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, 
                (255, 0, 255), 3)
    cv2.imshow("Image", img)
    #shows image 
    #"Image" is the name of the window
    #img is the image variable

    cv2.waitKey(1)