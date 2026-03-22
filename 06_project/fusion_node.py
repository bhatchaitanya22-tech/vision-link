import cv2
import numpy as np
from ultralytics import YOLO

# Known part dimensions (from job sheet — operator enters once)
CYLINDER_HEIGHT_MM = 50.0  # Z — change this to your actual part height
THETA = 0.0                # upright cylinders always 0

model = YOLO('/home/chaitanya-n-bhat/perception_map/06_project/best.pt')
H = np.load('/home/chaitanya-n-bhat/perception_map/data/calibration/homography_H.npy')

img_path = '/home/chaitanya-n-bhat/perception_map/data/raw/batch_03/IMG_20260321_155933.jpg'
img = cv2.imread(img_path)

h, w = img.shape[:2]
scale = 800 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))

results = model(img, conf=0.5, verbose=False)
boxes = results[0].boxes

print(f"Vision-Link Fusion Output")
print(f"{'='*40}")
print(f"Detected {len(boxes)} cylinder(s)\n")

for i, box in enumerate(boxes):
    cx, cy = map(int, box.xywh[0][:2].tolist())
    conf = float(box.conf[0])

    pixel = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    mm = cv2.perspectiveTransform(pixel, H)[0][0]

    print(f"Cylinder {i+1}:")
    print(f"  X:     {mm[0]:.1f} mm")
    print(f"  Y:     {mm[1]:.1f} mm")
    print(f"  Z:     {CYLINDER_HEIGHT_MM:.1f} mm")
    print(f"  Theta: {THETA:.1f} deg")
    print(f"  Conf:  {conf:.2f}")
    print()