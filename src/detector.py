from ultralytics import YOLO
import cv2

class VehicleDetector:
    """
    Vehicle detection module using YOLOv8. 
    
    This class loads a pre-trained YOLO model and provides a method to detect vehicles in video frames.
    """
    def __init__(self, model_path='yolov8n.pt'): # The 'yolov8n.pt' is a pre-trained model that can be used for vehicle detection.
        """
        Intializes the detector with a YOLO model.

        Args: 
            model_path (str): Path to the YOLO model weights. Default is 'yolov8n.pt'.
        """
        
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        """
        Detects vehicles in a single video frame.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            List[Dict]: A list of detections, where each detection is a dictionary containing:
                - 'label': The class label of the detected object (e.g., 'car', 'truck').
                - 'confidence': The confidence score of the detection.
                - 'box': A tuple of (x1, y1, x2, y2) representing the bounding box coordinates.
        """

        results = self.model(frame)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
        print(detections)
        return detections
