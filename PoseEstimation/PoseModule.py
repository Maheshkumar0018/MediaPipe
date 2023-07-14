import cv2
import mediapipe as mp
import time

class poseDetector:
    def __init__(self, mode=False, modelC=1, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.modelC =modelC
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelC, self.upBody, self.smooth,
                                     self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(rgb)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame,self.result.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return frame
    
    def findPosition(self, frame, draw=True):
        lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),10,(255,0,0),cv2.FILLED)
        return lmList
    
    def release(self):
        self.face_mesh = None

