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
FRAME_W, FRAME_H = 640, 480
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

# ================= YOLO =================
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

TARGET_OBJECTS = ["person", "dog", "cow", "cat", "elephant"]

CONF_THRESHOLD = 0.45       # ignore weak detections
SAFE_DISTANCE_CM = 120     # rail safety distance
BRANCH_AREA_MIN = 22000    # noise filter

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

# ================= AI HELPERS =================

def fog_level(frame):
    contrast = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()
    if contrast < 35:
        return "heavy"
    elif contrast < 55:
        return "light"
    return "clear"

def inside_track(box):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    return cv2.pointPolygonTest(TRACK_ROI, (cx, cy), False) >= 0

def estimate_camera_distance(box):
    x1,y1,x2,y2 = map(int, box)
    pixel_height = y2 - y1
    if pixel_height == 0:
        return 999
    focal_estimate = 700     # tuned constant
    real_height = 1.6       # avg human/animal height
    return round((real_height * focal_estimate) / pixel_height,1)

# ================= MAIN LOOP =================
print("SafeRail Optimized AI System Running...")

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        fog_status = fog_level(frame)

        results = model(frame, conf=CONF_THRESHOLD)
        annotated = results[0].plot()

        detected_objects = []
        closest_camera_dist = 999

        boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
        classes = results[0].boxes.cls.tolist() if results[0].boxes else []

        for cls, box in zip(classes, boxes):
            obj = CLASS_NAMES[int(cls)]

            x1,y1,x2,y2 = map(int, box)
            area = (x2-x1)*(y2-y1)

            # ---- MAIN OBJECTS ----
            if obj in TARGET_OBJECTS and inside_track(box):
                detected_objects.append(obj)

                cam_dist = estimate_camera_distance(box)
                closest_camera_dist = min(closest_camera_dist, cam_dist)

            # ---- TREE BRANCH ----
            if area > BRANCH_AREA_MIN and y2 > 350:
                detected_objects.append("tree_branch")

        detected_objects = list(set(detected_objects))

        # ================= ULTRASONIC =================
        try:
            ultrasonic_dist = round(sensor.distance * 100, 1)
        except:
            ultrasonic_dist = 999

        # ================= FUSED DISTANCE =================
        final_distance = min(closest_camera_dist, ultrasonic_dist)

        # ================= ESP DATA =================
        vibration = esp_data.get("vibration", 0)
        strain = esp_data.get("strain", 0)

        # ================= SMART INTRUSION LOGIC =================
        intrusion_detected = (
            len(detected_objects) > 0 and
            final_distance < SAFE_DISTANCE_CM
        )

        track_issue = vibration == 1 or strain > 40

        danger = intrusion_detected or track_issue

        # ================= ALERT (ONLY REAL THREATS) =================
        if intrusion_detected or track_issue:
            led.on()
            buzzer.on()
        else:
            led.off()
            buzzer.off()

        # ================= FINAL DATA =================
        final_data = {
            "objects": detected_objects,
            "distance_cm": round(final_distance,1),
            "fog": fog_status,
            "vibration": vibration,
            "strain": strain,
            "intrusion": intrusion_detected,
            "track_fault": track_issue,
            "danger": danger,
            "timestamp": time.time()
        }

        print(final_data)

        try:
            requests.put(FIREBASE_URL, json=final_data, timeout=1)
        except:
            pass

        # ================= LIVE FEED =================
        cv2.polylines(annotated, [TRACK_ROI], True, (0,255,0), 2)

        cv2.putText(annotated,f"Objects: {detected_objects}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.putText(annotated,f"Distance: {final_distance} cm",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.putText(annotated,f"Intrusion: {intrusion_detected}",(20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("SafeRail Optimized AI", annotated)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.03)

except KeyboardInterrupt:
    print("Stopping system...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    led.off()
    buzzer.off()
