import cv2
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.counter import LineCounter

def main():
    """
    Main execution script for the Smart Parking Vision System.

    This module orchestrates the full computer vision pipeline:

        Video Frame
            ↓
        Vehicle Detection (YOLOv8)
            ↓
        Multi-Object Tracking (DeepSORT)
            ↓
        Line-Crossing Counting Logic
            ↓
        Visualization & Display

    For each frame in the input video:
    - Vehicles are detected using a YOLO model.
    - Persistent IDs are assigned using DeepSORT tracking.
    - Vehicles are monitored for crossing a predefined virtual line.
    - Entry and exit counts are updated accordingly.
    - Bounding boxes, track IDs, and counters are rendered in real time.

    The video window closes when:
    - The end of the video is reached, or
    - The user presses the ESC key.

    Side Effects:
        - Reads video data from disk
        - Loads a trained YOLO model
        - Displays a real-time OpenCV window
        - Performs GPU/CPU inference depending on environment

    This file serves as the system entry point.
    """
    
    # Open video capture
    cap = cv2.VideoCapture("data\moving_cars_highway.mp4") # Sample video path
    
    detector = VehicleDetector() 
    tracker = VehicleTracker()
    counter = LineCounter()

    # Loop until video ends or user presses ESC
    while True:
        # Read a frame from the video capture
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
        counts = counter.update(tracks, frame.shape[0])

        # Car counting line (green line across the frame)
        cv2.line(frame, (0, 1060), (1919, 300), (0,255,0), 2)


        # Draw counters for in an out counts
        cv2.putText(frame,  f"IN: {counts['enter_count']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {counts['exit_count']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        # Draw centroid for each tracked object
        for track in tracks:
            x1, y1, x2, y2 = track['box']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Draw bounding boxes and labels for each tracked object
        for track in tracks:
            x1, y1, x2, y2 = track['box']
            track_id = track['id']
            label = track['label']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Track ID: {track_id}, Label: {label}, Box: ({x1}, {y1}, {x2}, {y2})")
            print(f"Tracks this frame: {len(tracks)}")


        # Creates a window and displays the frame with detected vehicles
        cv2.imshow("Vehicle Detection", frame)

        # Waits for 1 ms and checks if the ESC key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() # Entry point for standalone execution