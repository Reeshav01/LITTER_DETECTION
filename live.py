import cv2
from ultralytics import YOLO

URL = "http://192.168.1.5:8080/video"   # <- change to your phone's address
model = YOLO('best.pt')
cap = cv2.VideoCapture(URL)

i = 0
last = None
while True:
    ok, frame = cap.read()
    if not ok:
        print("No frame - check the address and WiFi")
        break
    if i % 2 == 0:                       # run the model on every 2nd frame
        r = model.predict(frame, conf=0.25, imgsz=480, verbose=False)[0]
        last = r.plot()
        cv2.putText(last, f"Litter: {len(r.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live litter detection (press q to quit)", last if last is not None else frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1

cap.release()
cv2.destroyAllWindows()