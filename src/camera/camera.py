import cv2
import time
from datetime import datetime
import os
from src.detection.detector import FaceDetector

class Camera:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            raise Exception("Could not open camera.")

        self.face_detector = FaceDetector()

        print("Camera Started Successfully.")

    def run(self):
        """Run the live camera feed."""

        prev_time = 0

        while True:

            # Read frame from webcam
            ret, frame = self.cap.read()

            if not ret:
                print("❌ Failed to read frame.")
                break

            # Mirror the camera
            frame = cv2.flip(frame, 1)
            
            detections = self.face_detector.detect_faces(frame)

            for face in detections:

                x, y, w, h = face["bbox"]

                confidence = face["confidence"]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                        
            # ---------------------------------------
            # Resolution
            # ---------------------------------------
            height, width, _ = frame.shape

            resolution = f"Resolution : {width} x {height}"

            cv2.putText(
                frame,
                resolution,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ---------------------------------------
            # FPS
            # ---------------------------------------
            current_time = time.time()

            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS : {int(fps)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # ---------------------------------------
            # Current Time
            # ---------------------------------------
            current_clock = datetime.now().strftime("%H:%M:%S")

            cv2.putText(
                frame,
                current_clock,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # ---------------------------------------
            # Keyboard Instructions
            # ---------------------------------------
            cv2.putText(
                frame,
                "Press S = Screenshot | Press Q = Quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ---------------------------------------
            # Display Window
            # ---------------------------------------
            cv2.imshow("AI Human Face Intelligence", frame)

            key = cv2.waitKey(1) & 0xFF

            # ---------------------------------------
            # Save Screenshot
            # ---------------------------------------
            if key == ord("s"):

                os.makedirs("assets/screenshots", exist_ok=True)

                filename = datetime.now().strftime("%Y%m%d_%H%M%S")

                filepath = f"assets/screenshots/{filename}.jpg"

                cv2.imwrite(filepath, frame)

                print(f"📸 Screenshot Saved : {filepath}")

            # ---------------------------------------
            # Quit
            # ---------------------------------------
            elif key == ord("q"):
                break

        self.stop()

    def stop(self):

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        print("Camera Closed.")