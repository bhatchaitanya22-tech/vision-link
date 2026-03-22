import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('/home/chaitanya-n-bhat/perception_map/'
             '06_project/best.pt')

H = np.load('/home/chaitanya-n-bhat/perception_map/'
            'data/calibration/homography_H.npy')

img_path = '/home/chaitanya-n-bhat/perception_map/data/raw/batch_03/IMG_20260321_155933.jpg'

img = cv2.imread(img_path)
h, w = img.shape[:2]
scale = 800 / max(h, w)
img = cv2.resize(img, (int(w*scale), int(h*scale)))

results = model(img, conf=0.5, verbose=False)
boxes = results[0].boxes

print(f"Detected {len(boxes)} cylinder(s)\n")

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    pixel_h = np.array([[cx], [cy], [1.0]])
    result = H @ pixel_h
    result = result / result[2]
    mm_x = result[0][0]
    mm_y = result[1][0]
    
    print(f"Cylinder {i+1}:")
    print(f"  Pixel center: ({cx:.0f}, {cy:.0f})")
    print(f"  Real world:   X={mm_x:.1f}mm, Y={mm_y:.1f}mm")
    print(f"  Confidence:   {conf:.2f}")
    
    cv2.circle(img, (int(cx), int(cy)), 8, (0,255,0), -1)
    cv2.putText(img,
                f"({mm_x:.0f},{mm_y:.0f})mm",
                (int(cx)+10, int(cy)-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2)
# At the end of detect_with_coords.py, replace your cv2.imshow line with:
cv2.namedWindow('Vision-Link Coordinates', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vision-Link Coordinates', 1000, 700)
cv2.imshow('Vision-Link Coordinates', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()