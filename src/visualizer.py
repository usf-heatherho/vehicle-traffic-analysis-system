import cv2

class FrameVisualizer:
    """
    Visualization module for rendering bounding boxes, track IDs, counting line, and entry/exit counts on video frames.

    This class provides a method to draw all necessary visual elements on each video frame based on the current tracking and counting state.
    """

    def draw(self, frame, tracks, enter_count, exit_count):
        # Car counting line (green line across the frame)
        cv2.line(frame, (0, 1060), (1919, 300), (0,255,0), 2)

        # Draw counters for in an out counts
        cv2.putText(frame,  f"INBOUND: {enter_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUTBOUND: {exit_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw centroid for each tracked object
        for track in tracks:
            x1, y1, x2, y2 = track['box']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            track_id = track['id']
            label = track['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame