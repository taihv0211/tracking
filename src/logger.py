# src/logger.py

"""Simple CSV-based logger for object tracking results."""

from datetime import datetime
import pandas as pd


class TrackingLogger:
    def __init__(self):
        self.records = []

    def log(self, track, frame_idx: int) -> None:
        """Store one tracking result."""
        x1, y1, x2, y2 = track["bbox"]
        self.records.append({
            "frame": frame_idx,
            "track_id": track["track_id"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self, csv_path: str) -> None:
        """Write all logged results to a CSV file."""
        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False)

