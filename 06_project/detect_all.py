import cv2
from ultralytics import YOLO
import os

model = YOLO('/home/chaitanya-n-bhat/perception_map/'
             '06_project/best.pt')

img_folder = '/home/chaitanya-n-bhat/perception_map/data/raw/batch_03/'
images = sorted([f for f in os.listdir(img_folder)
                 if f.lower().endswith(('.jpg', '.jpeg'))])

print(f"Running on {len(images)} images. Press any key for next. Press Q to quit.\n")

for fname in images:
    path = os.path.join(img_folder, fname)
    results = model(path, conf=0.5, verbose=False)
    
    boxes = results[0].boxes
    print(f"{fname}: {len(boxes)} cylinder(s) detected")
    
    result_img = results[0].plot()
    
    # resize for display
    h, w = result_img.shape[:2]
    scale = 800 / max(h, w)
    display = cv2.resize(result_img, (int(w*scale), int(h*scale)))
    
    cv2.imshow('Vision-Link Detection', display)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
print("Done.")