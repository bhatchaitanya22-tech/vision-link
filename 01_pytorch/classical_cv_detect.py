import cv2
import numpy as np
import os

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_pre = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    img_pre = cv2.GaussianBlur(img_pre, (3,3), 0)
    return img_pre

def detect_cylinders(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        return

    # resize for display — keep aspect ratio
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # preprocess
    img_pre = preprocess(img.copy())

    # grayscale + threshold
    gray = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    detected = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        # bounding box
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x,y),
                      (x+w_box, y+h_box), (0,255,0), 2)

        # center point
        cx, cy = x + w_box//2, y + h_box//2
        cv2.circle(result, (cx,cy), 6, (0,0,255), -1)

        # orientation — fitEllipse needs at least 5 points
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(result, ellipse, (255,0,0), 2)
            angle = ellipse[2]
            cv2.putText(
                result,
                f"({cx},{cy}) {angle:.0f}deg",
                (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 1
            )
            print(f"  Part {detected+1}: "
                  f"center=({cx},{cy})  "
                  f"theta={angle:.1f}deg  "
                  f"area={area:.0f}px")
            detected += 1

    print(f"  → {detected} object(s) detected")

    # show original vs result side by side
    combined = np.hstack([img, result])
    cv2.imshow('original | detected', combined)

    # also show threshold for debugging
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.imshow('threshold', thresh_color)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key

# --- run on all images ---
folder = os.path.expanduser(
    '~/perception_map/data/raw/batch_01'
)
images = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith(('.jpg','.jpeg','.png'))
])

print(f"Running on {len(images)} images.")
print("Press any key to go to next image. Press Q to quit.\n")

for fname in images:
    path = os.path.join(folder, fname)
    print(f"\n--- {fname} ---")
    key = detect_cylinders(path)
    if key == ord('q') or key == ord('Q'):
        print("Stopped by user.")
        break

print("\nDone.")