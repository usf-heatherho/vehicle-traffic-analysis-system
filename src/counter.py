class LineCounter:
    """
    Detects when tracked objects cross a virtual line in the frame and counts entries and exits.

    The counter maintains a history of tracked object positions to determine when they cross a predefined line in the frame. 
    It updates entry and exit counts based on the direction of crossing.
    """
    def __init__(self, line_position: float):
        """
        Initialize the LineCounter.

        Args:
            line_position (float): The vertical position of the counting line as a fraction of frame height (e.g., 0.5 for middle).
        """

        self.line_position = line_position  # Line at 50% of frame height
        self.track_history = {}
        self.counted_ids = set()
        self.enter_count = 0
        self.exit_count = 0

    def update(self, tracks, frame_height: int):
        """
        Update the counter with the current tracked objects.

        Args:
            tracks (list[dict]): List of tracked objects with 'id' and 'box' (x1, y1, x2, y2).
            frame_height (int): The height of the video frame to calculate line position.  

        Returns:            
            dict: Updated counts with 'enter_count' and 'exit_count'.
        """

        # A virtual line to check if tracked objects cross it.
        line_y = int(frame_height * self.line_position)

        for track in tracks:
            # Calculate the initial center of the bounding box for the current track
            track_id = track['id']
            x1, y1, x2, y2 = track['box']

            # original_center_x = (x1 + x2) / 2

            center_y = (y1 + y2) / 2

            # If car doesn't exist in track history
            if track_id not in self.track_history:
                self.track_history[track_id] = center_y
                continue
            
            # Update the track history with the current center y
            prev_center_y = self.track_history[track_id]

            # Only check for crossing if the track ID hasn't been counted yet
            if track_id not in self.counted_ids:
                # Crossing from above to below
                if prev_center_y < line_y <= center_y:  
                    self.exit_count += 1
                    self.counted_ids.add(track_id)

                # Crossing from below to above
                elif prev_center_y > line_y >= center_y:  
                    self.enter_count += 1
                    self.counted_ids.add(track_id)
            
            # Update track history with the current center y for the next frame
            self.track_history[track_id] = center_y

            print(f"Track ID: {track_id}, Center Y: {center_y}, Line Y: {line_y}, Enter Count: {self.enter_count}, Exit Count: {self.exit_count}")
        return {"enter_count": self.enter_count, "exit_count": self.exit_count, "line_y": line_y}

