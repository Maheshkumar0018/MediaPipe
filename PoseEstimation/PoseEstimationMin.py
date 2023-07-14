import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./PoseEstimation/videos/production_id_4841985 (2160p).mp4')

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

window_width = 960
window_height = 540
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', window_width, window_height)

pTime = 0

while(cap.isOpened()):
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame,result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            #print(cx, cy)
            cv2.circle(frame,(cx,cy),10,(255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(70,80),cv2.FONT_HERSHEY_PLAIN,
                7,(255,0,255),5)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()