# src/main.py

"""Entry point for the object tracking pipeline."""

import argparse
from pathlib import Path
import cv2

from detector import Detector
from tracker import Tracker
from logger import TrackingLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object tracking on a video")
    parser.add_argument("--video", default="../result/in/video2.mp4", help="Path to input video")
    parser.add_argument("--model", default="../best.pt", help="Path to YOLO model")
    parser.add_argument("--output", default="../result/out/out2.mp4", help="Path to save annotated video")
    parser.add_argument("--log", default="../result/out/tracking_log.csv", help="CSV file for saving tracking data")
    parser.add_argument("--record-all", action="store_true", help="Record every frame even if no active tracks")
    return parser.parse_args()


def run(video_path: Path, model_path: Path, output_path: Path, csv_path: Path, record_all: bool) -> None:
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    detector = Detector(str(model_path))
    tracker = Tracker()
    logger = TrackingLogger()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = detector.detect(frame)
        tracks, has_active = tracker.update(detections, frame, frame_idx)

        for track in tracks:
            logger.log(track, frame_idx)
            x1, y1, x2, y2 = track["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track['track_id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if record_all or has_active:
            out.write(frame)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.save(str(csv_path))


def main() -> None:
    args = parse_args()
    run(Path(args.video), Path(args.model), Path(args.output), Path(args.log), args.record_all)


if __name__ == "__main__":
    main()

