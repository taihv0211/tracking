# src/tracker.py

import cv2
import numpy as np
import math
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.deepsort = DeepSort(
            max_age=100,
            n_init=3,
            max_cosine_distance=0.2
        )
        self.kalman_filters = {}
        self.recent_lost_tracks = deque(maxlen=50)
        self.REID_FRAME_GAP = 30
        self.REID_DIST_THRESHOLD = 60

    def _init_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        return kalman

    def _euclidean_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections, frame, frame_idx):
        has_active_detection = False
        tracks_output = []
        trackers = self.deepsort.update_tracks(detections, frame=frame)

        for tracker in trackers:
            if not tracker.is_confirmed():
                continue

            track_id = tracker.track_id
            x1, y1, x2, y2 = map(int, tracker.to_tlbr())
            bbox = (x1, y1, x2, y2)

            if tracker.time_since_update == 0:
                center_now = ((x1 + x2) / 2, (y1 + y2) / 2)
                for lost in list(self.recent_lost_tracks):
                    predicted = lost["kalman"].predict()
                    predicted_center = (predicted[0], predicted[1])
                    dist = self._euclidean_distance(center_now, predicted_center)
                    if dist < self.REID_DIST_THRESHOLD and (frame_idx - lost["last_frame"]) <= self.REID_FRAME_GAP:
                        track_id = lost["track_id"]
                        break

            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = self._init_kalman_filter()

            center = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2]], dtype=np.float32)
            self.kalman_filters[track_id].correct(center)

            tracks_output.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2)
            })
            has_active_detection = True

        active_ids = [t.track_id for t in trackers if t.is_confirmed()]
        for tid in list(self.kalman_filters.keys()):
            if tid not in active_ids:
                self.recent_lost_tracks.append({
                    "track_id": tid,
                    "kalman": self.kalman_filters[tid],
                    "last_frame": frame_idx
                })

        return tracks_output, has_active_detection
