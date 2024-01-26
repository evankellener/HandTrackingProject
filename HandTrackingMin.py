import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, 
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.model_complexity = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        #hands object

        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity, self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils
        # function to all drawing between points

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #converts image to RGB

        self.results = self.hands.process(imgRGB)
        #processes the image

        #print(results.multi_hand_landmarks)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                #loops through all the hands
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                """
                for id, lm in enumerate(handLms.landmark):
                    #loops through all the landmarks in the form of id and landmark
                    h, w ,c= img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, "\nx: ", cx, "\ny: ", cy)
                    #draw 0th landmark
                    if id ==4:
                        cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                """

        return img
    

    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #loops through all the landmarks in the form of id and landmark
                h, w ,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, "\nx: ", cx, "\ny: ", cy)
                        #draw 0th landmark
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                    
        return lmList
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    #1 for webcam
    
    detector = handDetector()
    while True:  
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
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

if __name__ == "__main__":
    main()