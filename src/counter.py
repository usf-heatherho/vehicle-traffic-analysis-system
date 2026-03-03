class LineCounter:
    """
    Counts tracked objects that cross a predefined diagonal line
    using geometric side-of-line detection.

    This class maintains per-track state and detects when the centroid
    of a tracked object moves from one side of the counting line to the other.
    Crossing direction determines whether the object is counted as an entry
    or exit event.
    """

    def __init__(self):
        """
        Initialize the LineCounter with a fixed diagonal counting line.

        The line is defined by two endpoints in pixel coordinates.
        Objects are classified relative to this line using the sign
        of a 2D cross product.
        """

        # Define diagonal counting line (pixel coordinates)
        self.line_p1 = (0, 1060)
        self.line_p2 = (1919, 300)

        # Track ID -> previous side value
        self.track_history = {}

        # Track IDs already counted to prevent double counting
        self.counted_ids = set()

        self.enter_count = 0
        self.exit_count = 0

    def _compute_side(self, cx, cy):
        """
        Compute which side of the counting line a point lies on.

        Uses the 2D cross product to determine relative position.

        Args:
            cx (float): X-coordinate of object centroid.
            cy (float): Y-coordinate of object centroid.

        Returns:
            float: Signed value indicating which side of the line
                   the point lies on. Positive and negative values
                   represent opposite sides of the line.
        """

        x1, y1 = self.line_p1
        x2, y2 = self.line_p2

        return (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)

    def update(self, tracks):
        """
        Update entry and exit counts based on tracked object positions.

        Args:
            tracks (list of dict):
                Each track must contain:
                - "id": Unique track identifier
                - "box": Bounding box tuple (x1, y1, x2, y2)

        Returns:
            dict:
                {
                    "enter_count": int,
                    "exit_count": int
                }
        """

        for track in tracks:
            track_id = track["id"]
            x1, y1, x2, y2 = track["box"]

            # Compute centroid of bounding box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            current_side = self._compute_side(cx, cy)

            # If track did not exist before, initialize its history
            if track_id not in self.track_history:
                self.track_history[track_id] = current_side
                continue

            prev_side = self.track_history[track_id]

            # Only count once per track
            if track_id not in self.counted_ids:

                # Crossing occurs if sign changes
                if prev_side * current_side < 0:

                    # Direction classification
                    if prev_side > 0 and current_side < 0:
                        self.enter_count += 1
                    else:
                        self.exit_count += 1

                    self.counted_ids.add(track_id)

            # Update history for next frame
            self.track_history[track_id] = current_side

        return {
            "enter_count": self.enter_count,
            "exit_count": self.exit_count
        }