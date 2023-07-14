import cv2
import mediapipe as mp

class FaceMeshModule:
    
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = None

    def initialize(self):
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def findFaceMesh(self, image, draw=True):
        if self.face_mesh is None:
            raise RuntimeError("Module not initialized. Call initialize() first.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw:
                    # tessellation pattern for connecting the vertices
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # sets the color to red for the eye landmarks
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    # sets the color to green for the eye landmarks
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
                # faces with lanmarks
                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = image.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x,y])
            faces.append(face)

        return image, faces

    def release(self):
        self.face_mesh = None
