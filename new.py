import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import cv2
import time
import threading
from gpiozero import LED, Buzzer
from ultralytics import YOLO
import numpy as np

# =========================
# GPIO SETUP (YOUR PIN MAPPING)
# =========================
led_green = LED(5)      # GPIO 5  → Green LED
led_yellow = LED(6)     # GPIO 6  → Yellow LED (optional)
led_red = LED(13)       # GPIO 13 → Red LED
buzzer = Buzzer(18)     # GPIO 18 → Buzzer signal pin

# Initial state
led_green.on()
led_yellow.off()
led_red.off()
buzzer.off()

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("yolov8n.pt")

# High priority: Person + Animals (immediate danger)
PRIORITY_CLASSES = [
    "person", "dog", "cow", "horse", "sheep", "bear",
    "elephant", "cat", "zebra", "giraffe"
]

# Medium priority: non-living obstructions
NORMAL_CLASSES = [
    "car", "truck", "motorcycle", "bench",
    "backpack", "suitcase", "handbag"
]

# =========================
# RED LED BLINKING THREAD
# =========================
blink_red = False

def red_blinker():
    global blink_red
    while True:
        if blink_red:
            led_red.toggle()
            time.sleep(0.25)
        else:
            led_red.off()
            time.sleep(0.1)

threading.Thread(target=red_blinker, daemon=True).start()

# =========================
# DISTANCE ESTIMATION
# =========================
FOCAL_LENGTH = 650      # Calibrate for best accuracy
KNOWN_WIDTH = 0.5       # Approx width (meters) of animal/person

def estimate_distance(bbox_width_pixels):
    try:
        w = float(bbox_width_pixels)
        if w <= 0:
            return None
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w
        return round(distance, 2)
    except:
        return None


# =========================
# MAIN CAMERA LOOP
# =========================
def start_camera():
    global blink_red

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Camera started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            continue

        results = model(frame, stream=True)

        high_alert = False
        medium_alert = False
        min_distance = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0]

                # Convert tensor → float safely
                w = float(x2 - x1)

                distance = estimate_distance(w)

                # Track nearest object
                if distance is not None:
                    if min_distance is None or distance < min_distance:
                        min_distance = distance

                # HIGH PRIORITY
                if name in PRIORITY_CLASSES:
                    high_alert = True

                # MEDIUM PRIORITY
                elif name in NORMAL_CLASSES:
                    medium_alert = True

        # =========================
        # ALERT LOGIC
        # =========================
        if high_alert:
            print("🚨 HIGH ALERT: Person/Animal detected! Distance:", min_distance, "m")
            led_green.off()
            led_yellow.off()
            blink_red = True   # blinking red = danger
            buzzer.on()

        elif medium_alert:
            print("⚠ MEDIUM ALERT: Obstruction detected! Distance:", min_distance, "m")
            led_green.off()
            blink_red = False
            led_red.on()        # steady red = obstruction
            buzzer.on()

        else:
            # Safe track
            blink_red = False
            buzzer.off()
            led_red.off()
            led_green.on()

        # =========================
        # DISPLAY DISTANCE ON SCREEN
        # =========================
        if min_distance:
            cv2.putText(frame, f"Distance: {min_distance} m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("SafeRail Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
