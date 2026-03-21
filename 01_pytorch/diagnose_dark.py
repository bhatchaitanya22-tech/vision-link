import cv2
import numpy as np
import os

def diagnose(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Pixel value distribution:")
    for threshold in [50, 100, 150, 200, 240]:
        count = np.sum(gray > threshold)
        pct = count / gray.size * 100
        print(f"  pixels > {threshold}: {pct:.1f}%")

    results = []
    for t in [80, 120, 160, 200]:
        _, thresh = cv2.threshold(gray, t, 255,
                                   cv2.THRESH_BINARY)
        label = np.zeros_like(img)
        label[:] = (40, 40, 40)
        label[thresh == 255] = (255, 255, 255)
        cv2.putText(label, f"thresh={t}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,200,255), 2)
        results.append(label)

    top = np.hstack(results[:2])
    bot = np.hstack(results[2:])
    grid = np.vstack([top, bot])
    cv2.imshow('threshold comparison', grid)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

folder = os.path.expanduser(
    '~/perception_map/data/raw/batch_02'
)
images = sorted([f for f in os.listdir(folder)
                 if f.lower().endswith(
                     ('.jpg','.jpeg','.png'))])

print(f"Found {len(images)} images in batch_02\n")

for fname in images:
    print(f"\n--- {fname} ---")
    diagnose(os.path.join(folder, fname))