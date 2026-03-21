import cv2
import numpy as np

SQUARE_SIZE = 30
CORNERS = (8, 5)

img = cv2.imread(
    '/home/chaitanya-n-bhat/perception_map/'
    'data/calibration/chessboard_test.jpg'
)

if img is None:
    print("Image not found.")
    exit()

h, w = img.shape[:2]
scale = 800 / max(h, w)
img_display = cv2.resize(img, (int(w*scale), int(h*scale)))
img_gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)

found, corners = cv2.findChessboardCorners(
    img_gray, CORNERS,
    cv2.CALIB_CB_ADAPTIVE_THRESH +
    cv2.CALIB_CB_NORMALIZE_IMAGE
)

if not found:
    print("Chessboard not detected.")
    cv2.imshow('image', img_display)
    cv2.waitKey(0)
    exit()

print(f"Chessboard detected. {len(corners)} corners found.")

corners_refined = cv2.cornerSubPix(
    img_gray, corners, (11,11), (-1,-1),
    (cv2.TERM_CRITERIA_EPS +
     cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

real_points = []
for row in range(CORNERS[1]):
    for col in range(CORNERS[0]):
        real_points.append([col * SQUARE_SIZE,
                            row * SQUARE_SIZE])
real_points = np.array(real_points, dtype=np.float32)
pixel_points = corners_refined.reshape(-1, 2)

H, mask = cv2.findHomography(pixel_points, real_points)

print("\nHomography matrix H:")
print(np.round(H, 4))

# draw corners
img_interactive = img_display.copy()
cv2.drawChessboardCorners(
    img_interactive, CORNERS, corners_refined, found
)

# save H
np.save(
    '/home/chaitanya-n-bhat/perception_map/'
    'data/calibration/homography_H.npy', H
)
print("H matrix saved.")

# interactive — click to get mm coords
print("\nINTERACTIVE MODE")
print("Click anywhere on the image.")
print("Press Q to quit.")

H_matrix = H

def click_to_mm(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_h = np.array([[x, y, 1.0]]).T
        result = H_matrix @ pixel_h
        result = result / result[2]
        mm_x = result[0][0]
        mm_y = result[1][0]
        print(f"Pixel ({x}, {y}) → "
              f"({mm_x:.1f}mm, {mm_y:.1f}mm)")
        cv2.circle(param, (x,y), 5, (0,0,255), -1)
        cv2.putText(
            param,
            f"({mm_x:.0f},{mm_y:.0f})mm",
            (x+8, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0,0,255), 1
        )
        cv2.imshow('click to measure', param)

cv2.imshow('click to measure', img_interactive)
cv2.setMouseCallback(
    'click to measure', click_to_mm, img_interactive
)

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
print("Done.")