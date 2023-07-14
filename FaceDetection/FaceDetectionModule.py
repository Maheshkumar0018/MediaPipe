import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self,frame,draw=True):

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(rgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(frame,detection) # default function
                #print(detection.location_data.relative_bounding_box) 
                #print(detection.score)
                # manually draw the bounding box
                ih, iw, ic = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                #cv2.rectangle(frame,bbox,(255,0,255),5)
                if draw:
                    frame = self.fancyDraw(frame,bbox)
                    cv2.putText(frame,f'{int(detection.score[0] * 100)} %',
                            (bbox[0],bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,(255,0,255),2)
                
        return frame, bboxs
    
    def fancyDraw(self,frame,bbox,l=30,t=3,rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(frame,bbox,(255,0,255),rt)
        # top left x,y
        cv2.line(frame, (x,y), (x+l,y),(255,0,255),t)
        cv2.line(frame, (x,y), (x,y+l),(255,0,255),t)
        # top right x1,y
        cv2.line(frame, (x1,y), (x1-l,y),(255,0,255),t)
        cv2.line(frame, (x1,y), (x1,y+l),(255,0,255),t)
        # bottom left x,y
        cv2.line(frame, (x,y1), (x+l,y1),(255,0,255),t)
        cv2.line(frame, (x,y1), (x,y1-l),(255,0,255),t)
        # bottom right x1,y
        cv2.line(frame, (x1,y1), (x1-l,y1),(255,0,255),t)
        cv2.line(frame, (x1,y1), (x1,y1-l),(255,0,255),t)

        return frame
    
    def release(self):
        self.face_mesh = None
        
