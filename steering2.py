import cv2
import mediapipe as mp
import math
import pydirectinput
import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam input
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower res = better speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX
prev_time = 0  # For FPS calculation

with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)  # Mirror view
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        height, width, _ = image.shape
        wrist_coords = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Optional: comment this for performance boost
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                px, py = int(wrist.x * width), int(wrist.y * height)
                wrist_coords.append([px, py])

        if len(wrist_coords) == 0:
            cv2.putText(image, "Place hands in view", (30, 40), font, 0.8, (0, 0, 255), 2)

        elif len(wrist_coords) == 1:
            cv2.putText(image, "Reverse", (30, 40), font, 0.8, (0, 0, 255), 2)
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.keyDown('s')

        elif len(wrist_coords) == 2:
            x1, y1 = wrist_coords[0]
            x2, y2 = wrist_coords[1]
            xm, ym = (x1 + x2) // 2, (y1 + y2) // 2

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                angle_deg = 90
            else:
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)

            # Control logic
            pydirectinput.keyDown('w')
            pydirectinput.keyUp('s')

            if angle_deg < -25:
                pydirectinput.keyDown('a')
                pydirectinput.keyUp('d')
                cv2.putText(image, "Turn Left", (30, 40), font, 0.8, (255, 0, 0), 2)

            elif angle_deg > 25:
                pydirectinput.keyDown('d')
                pydirectinput.keyUp('a')
                cv2.putText(image, "Turn Right", (30, 40), font, 0.8, (0, 255, 0), 2)

            else:
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('d')
                cv2.putText(image, "Go Straight", (30, 40), font, 0.8, (0, 255, 255), 2)

        # FPS counter
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(image, f'FPS: {fps}', (10, 470), font, 0.6, (0, 255, 0), 2)

        cv2.imshow('Virtual Steering Wheel', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pydirectinput.keyUp('w')
pydirectinput.keyUp('a')
pydirectinput.keyUp('s')
pydirectinput.keyUp('d')
