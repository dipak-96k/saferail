import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import cv2
import time
import json
import threading
import paho.mqtt.client as mqtt
import pyrebase
from ultralytics import YOLO
from gpiozero import LED, Buzzer
from flask import Flask, Response
from flask_cors import CORS

# ================= FLASK =================
app = Flask(__name__)
CORS(app)

@app.after_request
def add_headers(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response

# ================= TARGET CLASSES =================
FAST_ALERT_CLASSES = ["elephant", "person"]
NORMAL_ALERT_CLASSES = ["dog", "cat", "cow"]

# ================= FIREBASE =================
firebaseConfig = {
  "apiKey": "AIzaSyCpkmnpS6Xby5k3smt0xWuO7L81T-SbQes",
  "authDomain": "saferail-vision.firebaseapp.com",
  "databaseURL": "https://saferail-vision-default-rtdb.firebaseio.com/",
  "storageBucket": "saferail-vision.appspot.com"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

user = auth.sign_in_with_email_and_password(
    "saferail@demo.com",
    "12345678"
)

# ================= MQTT =================
MQTT_BROKER = "localhost"
MQTT_TOPIC = "saferail/track"

esp_data = {}

def on_message(client, userdata, msg):
    global esp_data
    try:
        esp_data = json.loads(msg.payload.decode())
    except:
        pass

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883)
mqtt_client.subscribe(MQTT_TOPIC)
mqtt_client.on_message = on_message
mqtt_client.loop_start()

# ================= HARDWARE =================
led = LED(18)
buzzer = Buzzer(16)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================= YOLO =================
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

# ================= ALERT MEMORY =================
last_detect_time = 0
hold_time = 1.0
last_flash_time = 0
flash_state = False

# ================= VIDEO STREAM =================
def generate_frames():
    global last_detect_time, last_flash_time, flash_state

    last_firebase_update = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (480, 360))

        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        annotated = results[0].plot()

        detected_ids = []
        if results[0].boxes is not None:
            detected_ids = results[0].boxes.cls.tolist()

        detected_objs = [CLASS_NAMES[int(i)] for i in detected_ids]

        current_time = time.time()

        # ===== CLASS CHECK =====
        fast_detected = any(obj in FAST_ALERT_CLASSES for obj in detected_objs)
        normal_detected = any(obj in NORMAL_ALERT_CLASSES for obj in detected_objs)

        if fast_detected or normal_detected:
            last_detect_time = current_time

        alert_active = (current_time - last_detect_time) < hold_time

        # ===== FLASHING LOGIC =====
        if alert_active:

            if fast_detected:
                flash_interval = 0.2   # FAST flash (Elephant + Person)
            else:
                flash_interval = 0.5   # Normal flash (Dog/Cat/Cow)

            if current_time - last_flash_time > flash_interval:
                flash_state = not flash_state
                last_flash_time = current_time

            if flash_state:
                led.on()
                buzzer.on()
            else:
                led.off()
                buzzer.off()
        else:
            led.off()
            buzzer.off()

        # ===== FIREBASE UPDATE =====
        if current_time - last_firebase_update > 0.5:
            combined = {
                "wildlife_detected": detected_objs,
                "fast_alert": fast_detected,
                "normal_alert": normal_detected,
                "timestamp": current_time
            }

            try:
                db.child("saferail").child("live").set(
                    combined,
                    user['idToken']
                )
            except:
                pass

            last_firebase_update = current_time

        # ===== STREAM =====
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚆 SafeRail AI Running - Person Fast Mode Enabled")
    app.run(host='0.0.0.0', port=5000, threaded=True)
