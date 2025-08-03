# edge-object-tracking

This project demonstrates object tracking on UAV footage using YOLO for detection and DeepSORT with Kalman filter for tracking. Results can be saved as a video and as a CSV log.

---

## 📁 Project Folder Structure

```
project/
├── result/
│   ├── in/           # Input video files (e.g., video2.mp4)
│   └── out/          # Output video files and tracking logs
├── src/
│   ├── main.py       # Entry point
│   ├── detector.py   # YOLO detector wrapper
│   ├── tracker.py    # DeepSORT + Kalman + ReID tracking
│   └── logger.py     # CSV logger
├── best.pt           # YOLO model weights
├── requirements.txt  # Python dependencies
└── README.md         # Project instructions
```

---

## 🚀 How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare your folders**

Ensure `result/in/` contains the input video and `result/out/` exists for outputs.

3. **Run the tracker**

```bash
cd src
python main.py --video ../result/in/video2.mp4 \
               --model ../best.pt \
               --output ../result/out/out2.mp4 \
               --log ../result/out/tracking_log.csv
```

Pass `--record-all` if you want to record every frame even when no tracks are active.

4. **Output**

After running, the annotated video and CSV log will appear in `result/out/`.


## 🛠 Customize YOLO Layers

The project uses the `CustomYOLOv8` network defined in `src/yolo_model.py`.
This model is written entirely with PyTorch layers and does not rely on the
Ultralytics package.  Its architecture is kept simple so you can easily modify
the backbone or detection head.  Feel free to edit the layer list inside the
class to experiment with different designs.

Weights should be saved as a standard PyTorch `state_dict` (for example using
`torch.save(model.state_dict(), path)`).  Ultralytics checkpoints cannot be
loaded directly without conversion.  The `Detector` loads such a state dict and
runs inference using this custom network.
