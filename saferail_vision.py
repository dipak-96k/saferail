import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import cv2
import threading
import time
import json
import numpy as np
import requests
from gpiozero import LED, Buzzer, DistanceSensor
from ultralytics import YOLO
from flask import Flask, Response

# ================= FIREBASE =================
FIREBASE_URL = "https://saferail-ee182-default-rtdb.firebaseio.com/live.json"

# ================= DETECTABLE OBJECTS =================
DETECT_CLASSES = [
    "person", "cow", "dog", "cat", "elephant", "deer", "leopard"
]

# ================= HARDWARE =================
led = LED(17)
buzzer = Buzzer(18)
sensor = DistanceSensor(trigger=23, echo=24)

led.off()
buzzer.off()

# ================= CAMERA =================
FRAME_W, FRAME_H = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

# ================= YOLO =================
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names
CONF = 0.25

# ================= TRACK ROI =================
TRACK_ROI = np.array([
    [120, 480],
    [520, 480],
    [420, 260],
    [220, 260]
])

# ================= VIDEO STREAM =================
app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/video")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_stream():
    app.run(host="0.0.0.0", port=8000)

threading.Thread(target=run_stream, daemon=True).start()

# ================= HELPERS =================
def inside_track(box):
    x1,y1,x2,y2 = map(int, box)
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    return cv2.pointPolygonTest(TRACK_ROI, (cx, cy), False) >= 0

def fog_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c = gray.std()
    if c < 35: return "heavy"
    elif c < 55: return "light"
    return "clear"

# ================= MAIN LOOP =================
print("🚆 SafeRail AI + Firebase + Live Stream Running")

last_firebase = 0
intrusion_hold = 0

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        fog_status = fog_level(frame)

        results = model(frame, conf=CONF, verbose=False)

        boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
        classes = results[0].boxes.cls.tolist() if results[0].boxes else []

        detected = []

        for cls, box in zip(classes, boxes):
            obj = CLASS_NAMES[int(cls)]

            if obj in DETECT_CLASSES and inside_track(box):
                detected.append(obj)

            # ---- TREE BRANCH DETECTION (VISION HEURISTIC) ----
            x1,y1,x2,y2 = map(int, box)
            area = (x2-x1)*(y2-y1)
            if area > 20000 and y2 > 350:
                detected.append("tree_branch")

        detected = list(set(detected))

        # ================= DISTANCE =================
        try:
            distance_cm = round(sensor.distance * 100, 1)
        except:
            distance_cm = -1

        intrusion = len(detected) > 0

        # ================= ALERT CONTROL =================
        if intrusion:
            intrusion_hold = 6

        if intrusion_hold > 0:
            led.on()
            buzzer.on()
            intrusion_hold -= 1
        else:
            led.off()
            buzzer.off()

        # ================= FIREBASE DATA =================
        payload = {
            "objects": detected,
            "distance_cm": distance_cm,
            "fog": fog_status,
            "intrusion": intrusion,
            "timestamp": time.time()
        }

        print(payload)

        if time.time() - last_firebase > 0.5:
            try:
                requests.put(FIREBASE_URL, json=payload, timeout=0.5)
            except:
                pass
            last_firebase = time.time()

        # ================= DISPLAY =================
        cv2.polylines(frame, [TRACK_ROI], True, (0,255,0), 2)

        cv2.putText(frame, f"Objects: {detected}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Fog: {fog_status}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("SafeRail Live AI", frame)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.03)

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    led.off()
    buzzer.off()
