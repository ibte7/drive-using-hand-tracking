import cv2
import math
import time
import os
import mediapipe as mp
import pydirectinput as keyboard  

# Suppress TensorFlow/Mediapipe Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower res for performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX
prev_time = 0  # For FPS calculation

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2) as hands:

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Skipping empty frame.")
                continue

            # Mirror and convert for processing
            image = cv2.flip(image, 1)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imageHeight, imageWidth, _ = image.shape
            results = hands.process(imageRGB)

            image.flags.writeable = True
            co = []  # Wrist Coords

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draws fingers tracking (Comment it for performance)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    pixel = mp_drawing._normalized_to_pixel_coordinates(wrist.x, wrist.y, imageWidth, imageHeight)
                    if pixel:
                        co.append(list(pixel))

            # Handle no hands
            if len(co) == 0:
                cv2.putText(image, "Place hands in view", (50, 50), font, 0.9, (0, 0, 255), 2)
                continue

            # Handle both hands
            elif len(co) == 2:
                x1, y1 = co[0]
                x2, y2 = co[1]
                xm, ym = int((x1 + x2) / 2), int((y1 + y2) / 2)
                radius = 150

                try:
                    m = (y2 - y1) / (x2 - x1)
                    angle = math.degrees(math.atan(m))
                except ZeroDivisionError:
                    angle = 90  # Vertical

                cv2.circle(image, (xm, ym), radius, (0, 255, 0), 5)

                if (x2 > x1 and y2 > y1 and y2 - y1 > 65 and angle <= 24) or \
                   (x1 > x2 and y1 > y2 and y1 - y2 > 65 and angle <= 24):
                    direction = "Turn right"
                    keyboard.keyUp('a')
                    keyboard.keyUp('s')
                    keyboard.keyDown('w')
                    keyboard.keyDown('d')

                elif (x2 > x1 and y2 > y1 and y2 - y1 > 65 and 24 < angle < 89) or \
                     (x1 > x2 and y1 > y2 and y1 - y2 > 65 and 24 < angle < 89):
                    direction = "Full Turn right"
                    keyboard.keyUp('a')
                    keyboard.keyUp('s')
                    keyboard.keyDown('d')

                elif (x1 > x2 and y2 > y1 and y2 - y1 > 65 and angle >= -24) or \
                     (x2 > x1 and y1 > y2 and y1 - y2 > 65 and angle >= -24):
                    direction = "Turn left"
                    keyboard.keyUp('d')
                    keyboard.keyUp('s')
                    keyboard.keyDown('w')
                    keyboard.keyDown('a')

                elif (x1 > x2 and y2 > y1 and y2 - y1 > 65 and -90 < angle < -24) or \
                     (x2 > x1 and y1 > y2 and y1 - y2 > 65 and -90 < angle < -24):
                    direction = "Full Turn left"
                    keyboard.keyUp('d')
                    keyboard.keyUp('s')
                    keyboard.keyDown('a')

                else:
                    direction = "Keep straight"
                    keyboard.keyUp('a')
                    keyboard.keyUp('d')
                    keyboard.keyDown('w')

                cv2.putText(image, direction, (50, 50), font, 0.9, (0, 0, 0), 2)

            # Handle one hand — go backward
            elif len(co) == 1:
                direction = "Reversing"
                keyboard.keyUp('a')
                keyboard.keyUp('d')
                keyboard.keyUp('w')
                keyboard.keyDown('s')
                cv2.putText(image, direction, (50, 50), font, 0.9, (0, 0, 255), 2)

            # FPS counter
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(image, f'FPS: {fps}', (10, 470), font, 0.6, (0, 255, 0), 2)

            # Show the image
            cv2.imshow("Virtual Steering", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release keys and clean up
        for key in ['w', 'a', 's', 'd']:
            keyboard.keyUp(key)
        cap.release()
        cv2.destroyAllWindows()
