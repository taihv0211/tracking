# src/detector.py

import numpy as np
import torch

from yolo_model import CustomYOLOv8

class Detector:
    """Wrapper around a YOLO model for object detection."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model = CustomYOLOv8()
        self.device = device
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
        except (FileNotFoundError, Exception):
            # If no weights are found or format is incompatible we keep random initialization
            pass
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.2):
        """Run detection on a frame and return bounding boxes with confidences."""

        img = (
            torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        )
        img = img.to(self.device)
        preds = self.model(img)[0]
        preds = preds.permute(1, 2, 0).cpu().numpy()

        h, w, _ = frame.shape
        grid_h, grid_w, _ = preds.shape
        detections = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                tx, ty, tw, th, conf = preds[gy, gx, :5]
                if conf > conf_threshold:
                    bx = (gx + tx) / grid_w * w
                    by = (gy + ty) / grid_h * h
                    bw = np.exp(tw) * w / grid_w
                    bh = np.exp(th) * h / grid_h
                    detections.append(
                        ((bx - bw / 2, by - bh / 2, bw, bh), float(conf))
                    )

        return detections
