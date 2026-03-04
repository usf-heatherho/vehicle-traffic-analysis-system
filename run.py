import cv2
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.counter import LineCounter
from src.visualizer import FrameVisualizer

def main():
    """
    Entry point for the real-time vehicle traffic analytics system.

    Pipeline:
        Video Frame
            ↓
        Vehicle Detection (YOLO)
            ↓
        Multi-Object Tracking (DeepSORT)
            ↓
        Line Crossing Detection
            ↓
        Visualization
            ↓
        Display + Save Output
    """
    
    cap = cv2.VideoCapture("data/highway_traffic_15.mp4") # Sample video path
    
    detector = VehicleDetector() 
    tracker = VehicleTracker()
    counter = LineCounter()
    visualizer = FrameVisualizer()

    # Loop until video ends or user presses ESC
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        print(f"Processing frame of size: {width}x{height}, channels: {channels}")

        # Detect vehicles in the current frame
        detections = detector.detect_vehicles(frame)
        
        # Track detected vehicles across frames using DeepSORT
        tracks = tracker.update(detections, frame)

        # Update the line counter with current tracks and frame height
        counts = counter.update(tracks)

        #  Visualize the results on the frame and drawings
        frame = visualizer.draw(frame, tracks, counts['enter_count'], counts['exit_count'])

        # Creates a window and displays the frame with detected vehicles
        cv2.imshow("Traffic Analytics System", frame)

        # Waits for 1 ms and checks if the ESC key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release video capture and close windows
    input("Press Enter to exit...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() # Entry point for standalone execution