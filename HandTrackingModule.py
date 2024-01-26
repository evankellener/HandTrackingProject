import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
#1 for webcam

mpHands = mp.solutions.hands
#hands object

hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils
# function to all drawing between points

pTime = 0
cTime = 0

while True:  
    success, img = cap.read()
    #returns true if frame is read correctly

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #converts image to RGB

    results = hands.process(imgRGB)
    #processes the image

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #loops through all the hands
            for id, lm in enumerate(handLms.landmark):
                #loops through all the landmarks in the form of id and landmark
                h, w ,c= img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, "\nx: ", cx, "\ny: ", cy)
                #draw 0th landmark
                if id ==4:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

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