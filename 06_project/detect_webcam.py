import cv2
from ultralytics import YOLO

model = YOLO('/home/chaitanya-n-bhat/perception_map/06_project/best.pt')

cap = cv2.VideoCapture(0)
print("Press Q to quit.")

cv2.namedWindow('Vision-Link — Live Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vision-Link — Live Detection', 1000, 700)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    cv2.imshow('Vision-Link — Live Detection', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()