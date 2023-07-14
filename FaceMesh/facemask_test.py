import cv2
from FaceMaskModule_Cus import FaceMeshModule

face_mesh_module = FaceMeshModule()
face_mesh_module.initialize()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame, faces = face_mesh_module.findFaceMesh(frame)
    print(faces)
    cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and clean up
cap.release()
cv2.destroyAllWindows()

