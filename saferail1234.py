import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import cv2
import time
import json
import numpy as np
import requests
import threading
import paho.mqtt.client as mqtt
from gpiozero import LED, Buzzer, DistanceSensor
from ultralytics import YOLO

# ================= PERFORMANCE SETTINGS =================
FRAME_W, FRAME_H = 480, 360      # Lower resolution = faster
YOLO_INTERVAL = 3                # Run YOLO every 3 frames
CONF_THRESHOLD = 0.5
SAFE_DISTANCE_CM = 120

# ================= FIREBASE =================
FIREBASE_URL = "https://saferail-ee182-default-rtdb.firebaseio.com/live.json"

# ================= HARDWARE =================
led = LED(18)
buzzer = Buzzer(16)
sensor = DistanceSensor(trigger=23, echo=24)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

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

# ================= ROI =================
TRACK_ROI = np.array([
    [80, 360],
    [400, 360],
    [330, 200],
    [150, 200]
])

def inside_track(box):
    x1,y1,x2,y2 = map(int, box)
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    return cv2.pointPolygonTest(TRACK_ROI, (cx, cy), False) >= 0

# ================= DISTANCE ESTIMATION =================
def estimate_distance(box):
    x1,y1,x2,y2 = map(int, box)
    height = y2-y1
    if height <= 0:
        return 999
    focal = 650
    real_h = 1.6
    return (real_h * focal) / height

# ================= FIREBASE THREAD =================
latest_payload = {}

def firebase_sender():
    global latest_payload
    while True:
        try:
            if latest_payload:
                requests.put(FIREBASE_URL, json=latest_payload, timeout=0.5)
        except:
            pass
        time.sleep(0.5)

threading.Thread(target=firebase_sender, daemon=True).start()

# ================= MAIN LOOP =================
print("⚡ SafeRail High-Speed AI Running")

frame_count = 0
last_detection = []
smoothed_ultra = 999

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        detected_objects = []
        closest_cam_dist = 999

        # ---- Run YOLO every N frames ----
        if frame_count % YOLO_INTERVAL == 0:

            results = model.track(
                frame,
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False
            )

            boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
            classes = results[0].boxes.cls.tolist() if results[0].boxes else []

            for cls, box in zip(classes, boxes):
                obj = CLASS_NAMES[int(cls)]

                if obj in TARGET_OBJECTS and inside_track(box):
                    detected_objects.append(obj)

                    cam_dist = estimate_distance(box)
                    closest_cam_dist = min(closest_cam_dist, cam_dist)

            last_detection = detected_objects
        else:
            detected_objects = last_detection

        # ---- Ultrasonic smoothing ----
        try:
            raw_ultra = sensor.distance * 100
            smoothed_ultra = (0.7 * smoothed_ultra) + (0.3 * raw_ultra)
        except:
            pass

        final_distance = min(closest_cam_dist, smoothed_ultra)

        # ---- Smart Intrusion Logic ----
        intrusion = (
            len(detected_objects) > 0 and
            final_distance < SAFE_DISTANCE_CM
        )

        vibration = esp_data.get("vibration", 0)
        strain = esp_data.get("strain", 0)

        track_fault = vibration == 1 or strain > 40
        danger = intrusion or track_fault

        # ---- ALERT ONLY IF REAL THREAT ----
        if danger:
            led.on()
            buzzer.on()
        else:
            led.off()
            buzzer.off()

        # ---- Prepare Firebase payload ----
        latest_payload = {
            "objects": detected_objects,
            "distance_cm": round(final_distance,1),
            "intrusion": intrusion,
            "track_fault": track_fault,
            "danger": danger,
            "timestamp": time.time()
        }

        # ---- Display ----
        cv2.polylines(frame, [TRACK_ROI], True, (0,255,0), 2)

        cv2.putText(frame,f"Objects: {detected_objects}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.putText(frame,f"Distance: {round(final_distance,1)} cm",(10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame,f"Danger: {danger}",(10,85),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.imshow("SafeRail Turbo AI", frame)

        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    led.off()
    buzzer.off()
