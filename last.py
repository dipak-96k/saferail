import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import cv2
import time
import json
import threading
import numpy as np
import requests
import paho.mqtt.client as mqtt

from ultralytics import YOLO
from flask import Flask, Response
from gpiozero import LED, Buzzer, DistanceSensor

# ================= HARDWARE =================
led = LED(18)
buzzer = Buzzer(16)

ultrasonic = DistanceSensor(trigger=23, echo=24, max_distance=4)

# ================= FIREBASE =================
FIREBASE_URL = "https://saferail-ee182-default-rtdb.firebaseio.com/live.json"

# ================= CAMERA =================
FRAME_W, FRAME_H = 480, 360
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

# ================= YOLO =================
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

TARGET_OBJECTS = [
    "person", "dog", "cat", "cow", "horse", "sheep", "elephant"
]

CONF_THRESHOLD = 0.3

# ================= MQTT ESP DATA =================
esp_data = {
    "track_alignment": 0,
    "track_vibration": 0,
    "rail_stress": 0,
    "atmosphere_stress": 0,
    "fog_density": 0,
    "tilt_angle": 0,
    "safety_level": "LOW"
}

def on_message(client, userdata, msg):
    global esp_data
    try:
        esp_data = json.loads(msg.payload.decode())
    except:
        pass

mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883)
mqtt_client.subscribe("saferail/track")
mqtt_client.on_message = on_message
mqtt_client.loop_start()

# ================= ROI =================
TRACK_ROI = np.array([
    [0, FRAME_H],
    [FRAME_W, FRAME_H],
    [FRAME_W, 0],
    [0, 0]
])

def inside_track(box):
    x1,y1,x2,y2 = map(int, box)
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    return cv2.pointPolygonTest(TRACK_ROI, (cx, cy), False) >= 0

# ================= LIVE VIDEO SERVER =================
app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/video")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_stream():
    app.run(host="0.0.0.0", port=8000)

threading.Thread(target=run_stream, daemon=True).start()

# ================= MAIN LOOP =================

print("🚆 SafeRail AI Core Running")

last_firebase = 0
intrusion_hold = 0

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    # ===== YOLO DETECTION =====
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
    classes = results[0].boxes.cls.tolist() if results[0].boxes else []

    detected_objects = []

    for cls, box in zip(classes, boxes):
        name = CLASS_NAMES[int(cls)]

        if name in TARGET_OBJECTS and inside_track(box):
            detected_objects.append(name)

            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,name,(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    detected_objects = list(set(detected_objects))

    # ===== ULTRASONIC =====
    try:
        distance_cm = round(ultrasonic.distance * 100, 1)
    except:
        distance_cm = -1

    ultra_risk = distance_cm > 0 and distance_cm < 80

    # ===== INTRUSION LOGIC =====
    intrusion = len(detected_objects) > 0 or ultra_risk

    # ===== ALERT =====
    if intrusion:
        intrusion_hold = 6

    if intrusion_hold > 0:
        led.on()
        buzzer.on()
        intrusion_hold -= 1
    else:
        led.off()
        buzzer.off()

    # ===== AI + SENSOR FUSION =====
    safety_level = esp_data.get("safety_level", "LOW")

    danger = intrusion or safety_level == "HIGH"

    payload = {
        "objects_detected": detected_objects,
        "intrusion": intrusion,

        "distance_cm": distance_cm,
        "ultrasonic_risk": ultra_risk,

        "track_alignment": esp_data.get("track_alignment", 0),
        "track_vibration": esp_data.get("track_vibration", 0),
        "rail_stress": esp_data.get("rail_stress", 0),
        "atmosphere_stress": esp_data.get("atmosphere_stress", 0),
        "fog_density": esp_data.get("fog_density", 0),
        "tilt_angle": esp_data.get("tilt_angle", 0),
        "safety_level": safety_level,

        "danger": danger,
        "timestamp": time.time()
    }

    print(payload)

    # ===== FIREBASE PUSH =====
    if time.time() - last_firebase > 0.5:
        try:
            requests.put(FIREBASE_URL, json=payload, timeout=0.5)
        except:
            pass
        last_firebase = time.time()

    # ===== LOCAL VIEW =====
    cv2.imshow("SafeRail AI Live", frame)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
led.off()
buzzer.off()
