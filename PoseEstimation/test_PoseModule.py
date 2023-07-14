import cv2
import mediapipe as mp
import time
import PoseModule as pm

  
cap = cv2.VideoCapture('./PoseEstimation/videos/11.mp4')

window_width = 960
window_height = 540
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', window_width, window_height)
detector = pm.poseDetector()
pTime = 0
while(cap.isOpened()):
    _, frame = cap.read()
    frame = detector.findPose(frame)
    lmList = detector.findPosition(frame)
    cv2.circle(frame, (lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
    print(lmList)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(70,80),cv2.FONT_HERSHEY_PLAIN,
                    7,(255,0,255),5)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break