import cv2


class Camera:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise Exception("Could not open camera.")

        print("Camera Started Successfully.")

    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read frame.")
                break

            cv2.imshow("AI Human Face Intelligence", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

        self.stop()

    def stop(self):

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        print("Camera Closed.")