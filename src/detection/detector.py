# ==========================================
# Imports
# ==========================================

import cv2
import mediapipe as mp


# ==========================================
# Face Detector Class
# ==========================================

class FaceDetector:

    def __init__(self):

        self.mp_face = mp.solutions.face_detection

        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    # --------------------------------------

    def detect_faces(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb)

        detections = []

        if results.detections:

            height, width, _ = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                confidence = detection.score[0]

                detections.append({
                    "bbox": (x, y, w, h),
                    "confidence": confidence
                })

        return detections