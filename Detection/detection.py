import cv2
import time
import mediapipe as mp

class FaceMeshHandPoseDetector:
    def __init__(self, mode=False, max_hands=2, model_c=1, detection_con=0.5, track_con=0.5, up_body=False, smooth=True):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.face_mesh = None
        self.hands = None
        self.pose = None

        self.mode = mode
        self.max_hands = max_hands
        self.model_c = model_c
        self.detection_con = detection_con
        self.track_con = track_con
        self.up_body = up_body
        self.smooth = smooth

    def initialize(self):
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands,
            self.model_c,
            self.detection_con,
            self.track_con
        )

        self.pose = self.mp_pose.Pose(
            self.mode,
            self.model_c,
            self.up_body,
            self.smooth,
            self.detection_con,
            self.track_con
        )

    def find_face_mesh(self, image, draw=True):
        if self.face_mesh is None:
            raise RuntimeError("FaceMesh not initialized. Call initialize() first.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    self.mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    self.mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = image_bgr.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)

        return image_bgr, faces

    def find_hands(self, image, draw=True):
        if self.hands is None:
            raise RuntimeError("Hands not initialized. Call initialize() first.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS
                    )

        return image_bgr

    def find_hand_positions(self, image, hand_no=0, draw=True):
        if self.hands is None:
            raise RuntimeError("Hands not initialized. Call initialize() first.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        hand_positions = []

        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = image_bgr.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_positions.append([id, cx, cy])

                if draw:
                    cv2.circle(image_bgr, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return image_bgr, hand_positions

    def find_pose(self, frame, draw=True):
        if self.pose is None:
            raise RuntimeError("Pose not initialized. Call initialize() first.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame

    def find_pose_positions(self, frame, draw=True):
        if self.pose is None:
            raise RuntimeError("Pose not initialized. Call initialize() first.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        pose_positions = []

        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                pose_positions.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return frame, pose_positions

    def release(self):
        self.face_mesh = None
        self.hands = None
        self.pose = None
