import os
import cv2

folder = os.path.expanduser(
    '~/perception_map/data/raw/batch_03'
)

images = [f for f in os.listdir(folder) 
          if f.lower().endswith(
              ('.jpg','.jpeg','.png')
          )]

print(f"Total images: {len(images)}")

for fname in images[:5]:
    path = os.path.join(folder, fname)
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        print(f"{fname}: {w}x{h}")
    else:
        print(f"{fname}: FAILED TO LOAD")