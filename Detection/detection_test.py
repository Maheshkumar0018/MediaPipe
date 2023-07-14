import cv2
from detection import FaceMeshHandPoseDetector


detector = FaceMeshHandPoseDetector()
detector.initialize()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform face mesh detection, hand tracking, and pose estimation
    annotated_frame, _ = detector.find_face_mesh(frame)
    annotated_frame = detector.find_hands(annotated_frame)
    annotated_frame = detector.find_pose(annotated_frame)

    cv2.imshow('Face Mesh, Hands, and Pose', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

detector.release()


