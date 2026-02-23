import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import cv2
import time
import json
import numpy as np
import requests
import paho.mqtt.client as mqtt
from gpiozero import LED, Buzzer, DistanceSensor
from ultralytics import YOLO

# ================= FIREBASE =================
FIREBASE_URL = "https://saferail-ee182-default-rtdb.firebaseio.com/live.json"

# ================= HARDWARE =================
led = LED(18)
buzzer = Buzzer(16)
sensor = DistanceSensor(trigger=23, echo=24)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ================= YOLO =================
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

TARGET_OBJECTS = ["person", "dog", "cow", "cat", "elephant"]

# ================= MQTT =================
esp_data = {}

def on_message(client, userdata, msg):
    global esp_data
    try:
        esp_data = json.loads(msg.payload.decode())
    except:
        pass

client = mqtt.Client()
client.connect("localhost", 1883)
client.subscribe("saferail/track")
client.on_message = on_message
client.loop_start()

# ================= TRACK ROI =================
TRACK_ROI = np.array([
    [120, 480],
    [520, 480],
    [420, 260],
    [220, 260]
])

# ================= FOG ESTIMATION =================
def fog_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    if contrast < 35:
        return "heavy"
    elif contrast < 55:
        return "light"
    else:
        return "clear"

# ================= ROI CHECK =================
def inside_track(box):
    x1, y1, x2, y2 = map(int, box)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cv2.pointPolygonTest(TRACK_ROI, (cx, cy), False) >= 0

# ================= MAIN LOOP =================
print("SafeRail Edge AI System Running...")

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        fog_status = fog_level(frame)

        results = model(frame)
        annotated = results[0].plot()

        detected = []

        boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
        classes = results[0].boxes.cls.tolist() if results[0].boxes else []

        for cls, box in zip(classes, boxes):
            obj = CLASS_NAMES[int(cls)]

            # Main objects
            if obj in TARGET_OBJECTS and inside_track(box):
                detected.append(obj)

            # Tree branch heuristic (large object low in frame)
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            if area > 20000 and y2 > 350:
                detected.append("tree_branch")

        detected = list(set(detected))

        # ================= ULTRASONIC =================
        try:
            distance_cm = round(sensor.distance * 100, 1)
        except:
            distance_cm = -1

        # ================= ESP DATA =================
        vibration = esp_data.get("vibration", 0)
        fog_sensor = esp_data.get("fog", 0)
        strain = esp_data.get("strain", 0)
        ax = esp_data.get("ax", 0)

        # ================= FINAL DECISION =================
        intrusion = len(detected) > 0
        close_object = 0 < distance_cm < 120
        track_fault = vibration == 1 or strain > 40

        danger = intrusion or close_object or track_fault or fog_status == "heavy"

        # ================= ALERT =================
        if danger:
            led.on()
            buzzer.on()
        else:
            led.off()
            buzzer.off()

        # ================= FINAL DATA =================
        final_data = {
            "detected_objects": detected,
            "distance_cm": distance_cm,
            "fog_camera": fog_status,
            "fog_sensor": fog_sensor,
            "vibration": vibration,
            "strain": strain,
            "acceleration": ax,
            "danger": danger,
            "timestamp": time.time()
        }

        print(final_data)

        # ================= SEND TO FIREBASE =================
        try:
            requests.put(FIREBASE_URL, json=final_data, timeout=1)
        except Exception as e:
            print("Firebase error:", e)

        # ================= LIVE FEED =================
        cv2.polylines(annotated, [TRACK_ROI], True, (0,255,0), 2)

        cv2.putText(annotated, f"Objects: {detected}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(annotated, f"Distance: {distance_cm} cm", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(annotated, f"Fog: {fog_status}", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("SafeRail Live AI", annotated)

        if cv2.waitKey(1) == 27:   # ESC to quit
            break

        time.sleep(0.04)

except KeyboardInterrupt:
    print("Stopping system...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    led.off()
    buzzer.off()
