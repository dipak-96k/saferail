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
led_green = LED(5)
led_yellow = LED(6)
led_red = LED(13)
buzzer = Buzzer(18)

led_green.on()
led_yellow.off()
led_red.off()
buzzer.off()

# =========================
# YOLO MODEL
# =========================
model = YOLO("yolov8n.pt")

# Hard species filtering
ALLOWED_SPECIES = ["cow", "dog", "elephant", "person"]

# Stable detection (reduce flicker)
stable_count = 0
STABLE_REQUIRED = 3

# =========================
# MOTION DETECTOR
# =========================
motion_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=25, detectShadows=False
)

# =========================
# REGION OF INTEREST (ROI)
# =========================
# Only detect in bottom 60% of frame
ROI_TOP = 0.40       # adjust based on camera angle


# =========================
# BLINKING RED LED THREAD
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
FOCAL_LENGTH = 650
KNOWN_WIDTH = 0.5   # approx width of dog/cow/person

def estimate_distance(w):
    try:
        w = float(w)
        if w <= 0:
            return None
        return round((KNOWN_WIDTH * FOCAL_LENGTH) / w, 2)
    except:
        return None

# =========================
# MAIN CAMERA LOOP
# =========================
def start_camera():
    global blink_red, stable_count

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Camera started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape

        # ROI Crop
        roi_start = int(H * ROI_TOP)
        roi = frame[roi_start:H, :]

        # =========================
        # MOTION DETECTION
        # =========================
        fgmask = motion_subtractor.apply(roi)
        motion_pixels = cv2.countNonZero(fgmask)

        motion_detected = motion_pixels > 1500   # prevents false static detection

        # =========================
        # YOLO DETECTION ON ROI ONLY
        # =========================
        results = model(roi, stream=True, conf=0.65, iou=0.45)

        detected = False
        min_distance = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                conf = float(box.conf[0])

                # HARD SPECIES FILTER
                if name not in ALLOWED_SPECIES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                w = float(x2 - x1)
                h = float(y2 - y1)

                # Ignore tiny box
                if w < 60 or h < 60:
                    continue

                # Must have motion
                if not motion_detected:
                    continue

                # Distance
                distance = estimate_distance(w)
                if distance is not None:
                    if min_distance is None or distance < min_distance:
                        min_distance = distance

                detected = True

        # =========================
        # STABILITY FILTER (3 FRAMES)
        # =========================
        if detected:
            stable_count += 1
        else:
            stable_count = 0

        # =========================
        # ALERT LOGIC
        # =========================
        if stable_count >= STABLE_REQUIRED:
            print("🚨 REAL DETECTION —", name, "Distance:", min_distance, "m")
            blink_red = True
            buzzer.on()
            led_green.off()
            led_yellow.off()
        else:
            blink_red = False
            buzzer.off()
            led_red.off()
            led_green.on()

        # =========================
        # OVERLAY INFORMATION
        # =========================
        if min_distance:
            cv2.putText(frame, f"Distance: {min_distance} m",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # Draw ROI line
        cv2.line(frame, (0, roi_start), (W, roi_start), (0, 255, 255), 2)
        cv2.putText(frame, "ROI START", (10, roi_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("SafeRail Vision (Filtered)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
