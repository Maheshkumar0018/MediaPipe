import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./FaceMesh/videos/1.mp4')

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

pTime = 0
window_width = 960
window_height = 540
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', window_width, window_height)

while(cap.isOpened()):
    _, frame = cap.read()

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    print(results)
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame,faceLm,mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec,drawSpec)
            for id, lm in enumerate(faceLm.landmark):
                ih, iw, ic = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)


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
