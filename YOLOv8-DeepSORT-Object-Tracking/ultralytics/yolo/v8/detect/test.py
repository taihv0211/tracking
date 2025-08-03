# test_yolov8_image.py
# filepath: test_yolov8_image.py

from ultralytics import YOLO
import cv2

# Load model đã train
model = YOLO('/home/trongan93/taihv/YOLOv8-DeepSORT-Object-Tracking/best.pt')  # Thay bằng đường dẫn model của bạn

# Đọc ảnh đầu vào
img_path = '/home/trongan93/taihv/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/data/datasets/UAVDT/images/val/M0101_000062.jpg'   # Thay bằng đường dẫn ảnh của bạn
img = cv2.imread(img_path)

# Dự đoán
results = model(img)

# Vẽ bounding box lên ảnh
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()
    for box, conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# Lưu ảnh kết quả
cv2.imwrite('output.jpg', img)
print('Đã lưu ảnh kết quả: output.jpg')