import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('/home/chaitanya-n-bhat/perception_map/06_project/best.pt')

img_path = '/home/chaitanya-n-bhat/perception_map/data/raw/batch_03/IMG_20260321_155933.jpg'
img = cv2.imread(img_path)

h, w = img.shape[:2]
scale = 800 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))

results = model(img, conf=0.5, verbose=False)
boxes = results[0].boxes

print(f"Detected {len(boxes)} cylinder(s)\n")

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cx, cy = map(int, box.xywh[0][:2].tolist())

    crop = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"Cylinder {i+1}: no contour found")
        continue

    largest = max(contours, key=cv2.contourArea)

    theta = None
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        theta = ellipse[2]
        # Draw ellipse on original image (offset back to original coords)
        center = (int(x1 + ellipse[0][0]), int(y1 + ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        angle = ellipse[2]
        cv2.ellipse(img, center, axes, angle, 0, 360, (0, 255, 255), 2)

    print(f"Cylinder {i+1}:")
    print(f"  Center: ({cx}, {cy})")
    print(f"  Theta:  {theta:.1f} degrees" if theta is not None else "  Theta: could not estimate")
    print()

cv2.namedWindow('Vision-Link Pose', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vision-Link Pose', 1000, 700)
cv2.imshow('Vision-Link Pose', img)
cv2.waitKey(0)
cv2.destroyAllWindows()