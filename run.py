import cv2
from app.detection.detector import VehicleDetector

def main():
    """
    Simple demo loop that opens a video file and runs the vehicle
    detector on each frame.

    The video source is currently hard‑coded to
    "moving_cars_highway.mp4". A VehicleDetector class object is
    instantiated and, for every frame read from the capture, its
    detect_vehicles() method is called.  Detected bounding boxes are
    drawn in green on the frame and the annotated frame is shown in an
    OpenCV window titled “Vehicle Detection”.

    The loop terminates when the end of the video is reached or when the
    user presses the ESC key.  Before returning, the function releases
    the cv2.VideoCapture class and destroys any OpenCV windows it has
    created.

    Side effects
    ----------
    - reads from disk
    - displays an OpenCV window
    - blocks until video end or user interaction
    """
    
    # Open video capture
    cap = cv2.VideoCapture("moving_cars_highway.mp4") # Sample video path
    
    detector = VehicleDetector() 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_vehicles(frame)

        for det in detections:
            x1, y1, x2, y2 = det['box']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27: # Press 'ESC' to exit
            break

        # Release video capture and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()