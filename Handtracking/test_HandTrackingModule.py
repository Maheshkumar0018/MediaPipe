import cv2
import mediapipe as mp
import time
import HandTrackingModule as ht

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = ht.handDetector()
while True:
    _, frame = cap.read()
    frame = detector.findHands(frame)
    lmLst = detector.findPosition(frame,draw=False)
    if len(lmLst) != 0:
        print(lmLst)
        #print(lmLst[18]) 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break