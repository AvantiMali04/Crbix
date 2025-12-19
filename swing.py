import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

prev_x = {"left": None, "right": None}
direction = {"left": None, "right": None}
swings = {"left": 0, "right": 0}

THRESHOLD = 50  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    wrists = {}
    if results and results[0].keypoints is not None:
        kpts = results[0].keypoints.xy[0]
        confs = results[0].keypoints.conf[0]

        if confs[9] > 0.5:
            wrists["left"] = kpts[9]

        if confs[10] > 0.5:
            wrists["right"] = kpts[10]

    for hand, coord in wrists.items():
        x = int(coord[0])

        if prev_x[hand] is not None:
            if x > prev_x[hand] + THRESHOLD and direction[hand] != "right":
                swings[hand] += 1
                direction[hand] = "right"
            elif x < prev_x[hand] - THRESHOLD and direction[hand] != "left":
                swings[hand] += 1
                direction[hand] = "left"

        prev_x[hand] = x

    cv2.imshow("Hand Swing Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

total_swings = swings["left"] + swings["right"]
print("\nFinal Counts → Left:", swings["left"], 
      "| Right:", swings["right"], 
      "| Total:", total_swings)
