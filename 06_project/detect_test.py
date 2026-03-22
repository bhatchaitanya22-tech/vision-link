import cv2
from ultralytics import YOLO
import os

# load your trained model
model = YOLO('/home/chaitanya-n-bhat/perception_map/06_project/best.pt')

# pick one image from your dataset
img_path = '/home/chaitanya-n-bhat/perception_map/data/raw/batch_03/IMG_20260321_155933.jpg'

# run inference
results = model(img_path, conf=0.5)

# draw results
for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} cylinder(s)")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        print(f"  Center: ({cx:.0f}, {cy:.0f}) pixels  "
              f"Confidence: {conf:.2f}")

# save annotated image
result_img = results[0].plot()
output_path = '/home/chaitanya-n-bhat/perception_map/06_project/detection_result.jpg'
cv2.imwrite(output_path, result_img)
print(f"\nResult saved to: {output_path}")