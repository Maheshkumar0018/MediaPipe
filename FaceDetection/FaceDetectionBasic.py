import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./FaceDetection/videos/7.mp4')

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

window_width = 960
window_height = 540
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', window_width, window_height)
pTime = 0
while(cap.isOpened()):
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(frame,detection) # default function
            #print(detection.location_data.relative_bounding_box) 
            #print(detection.score)
            # manually draw the bounding box
            ih, iw, ic = frame.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame,bbox,(255,0,255),5)
            cv2.putText(frame,f'{int(detection.score[0] * 100)} %',
                        (bbox[0],bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                    7,(255,0,255),5)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame,f'FPS: {int(fps)}',(70,80),cv2.FONT_HERSHEY_PLAIN,
                    7,(255,0,255),5)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()