import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import os
import config

# Landmark names
LANDMARK_NAMES = {
    0:  "NOSE",
    7:  "LEFT_EAR",
    8:  "RIGHT_EAR",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")


class PoseDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Download it with:\n"
                "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'"
                " -OutFile pose_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def detect(self, frame):
        """
        Process a BGR frame.
        Returns (annotated_frame, landmarks_dict or None).
        landmarks_dict keys: NOSE, LEFT_EAR, RIGHT_EAR,
                             LEFT_SHOULDER, RIGHT_SHOULDER,
                             LEFT_HIP, RIGHT_HIP, etc.
        Each value: (x, y, z, visibility) -- x,y,z normalized [0,1]
        """
        self._frame_timestamp_ms += 33  # ~30 FPS

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms)

        landmarks_dict = None
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks_dict = self._extract(result.pose_landmarks[0])
            frame = self._draw(frame, result.pose_landmarks[0])

        return frame, landmarks_dict

    def _extract(self, landmarks):
        out = {}
        for idx, name in LANDMARK_NAMES.items():
            lm = landmarks[idx]
            out[name] = (lm.x, lm.y, lm.z, lm.visibility)
        return out

    def _draw(self, frame, landmarks):
        h, w = frame.shape[:2]
        # Draw skeleton connections (subset)
        connections = [
            (11, 12),  # shoulders
            (11, 23), (12, 24),  # torso sides
            (23, 24),  # hips
            (7, 11), (8, 12),  # ear to shoulder
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
        ]
        pts = {}
        for idx in LANDMARK_NAMES:
            lm = landmarks[idx]
            pts[idx] = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pts[idx], 5, (0, 255, 0), -1)

        for a, b in connections:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
        return frame

    def close(self):
        self.landmarker.close()


# Smoke test
if __name__ == "__main__":
    detector = PoseDetector()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    print("Pose test - press q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, landmarks = detector.detect(frame)
        msg = "Pose OK" if landmarks else "No pose"
        color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if landmarks:
            cv2.putText(frame,
                f"Trunk: {landmarks['LEFT_SHOULDER'][:2]}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("Pose Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    detector.close()