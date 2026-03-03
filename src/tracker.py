from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    """
    Encapsulates the DeepSORT tracking algorithm for tracking vehicles
    across video frames with persistent IDs.
    """

    def __init__(self, max_age=15, n_init=2, max_iou_distance=0.7):
        """
        Initialize the DeepSORT tracker.

        Args:
            max_age (int): Maximum number of frames to keep a track alive
                           without detections before deletion.
            n_init (int): Number of consecutive detections required before
                          a track is confirmed.
            max_iou_distance (float): IOU threshold for matching detections
                                      to existing tracks.
        """
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance)

    def update(self, detections, frame):
        """
        Update tracker with detections from the current frame.

        Args:
            detections (list[dict]): Each detection must contain:
                - 'label': str
                - 'confidence': float
                - 'box': (x1, y1, x2, y2)
            frame (np.ndarray): The current video frame.

        Returns:
            list[dict]: Tracked objects containing:
                - 'id'
                - 'label'
                - 'box' (x1, y1, x2, y2)
        """

        # Convert detections into DeepSORT expected format:
        # ([x, y, w, h], confidence, class_name)
        deep_sort_detections = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            w = x2 - x1
            h = y2 - y1

            deep_sort_detections.append(
                ([x1, y1, w, h], det["confidence"], det["label"])
            )

        # Update tracker
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

        output = []

        for track in tracks:

            # Only use confirmed tracks updated this frame
            if not track.is_confirmed():
                continue

            if track.time_since_update > 0:
                continue

            bbox = track.to_tlbr()  # (x1, y1, x2, y2)

            output.append({
                "id": track.track_id,
                "label": track.det_class, 
                "box": (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                )
            })

        print(output)
        return output
    